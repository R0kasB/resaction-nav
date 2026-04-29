import torch
import numpy as np
import torch.nn.functional as F
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from transformers import AutoModel, AutoProcessor
import random
import math
    
# to read: https://allenai.github.io/ai2thor-v2.1.0-documentation/actions/initialization
# https://gymnasium.farama.org/introduction/basic_usage/

class BaseAgent:
    def __init__(
        self,
        target_object_types = None,
        base_resolution = (224, 224),
        success_distance = 0.5,
        #base_downgrade: int = 4,
        max_steps = 200,
        sensing_cost = 0.02,
        max_sensing_budget = 5,
        move_magnitude = 0.25,
        rotate_degrees = 45.0,
        look_degrees = 15.0,
        visibility_distance = 1.5,
        success_distance = 1.0,
        encoder_name: str = "facebook/dinov2-base",
        device = 'cuda',
        seed = None,
        step_penalty = 0.001,
        distance_scale = ...,
        sense_penalty = ...,
        fail_penalty = ...,
        bump_penalty = ...,
        success_reward = ...,
    ):
        self.target_object_types = target_object_types
        self.success_distance = success_distance
        self.base_resolution = base_resolution
        #start with the lowest resolution (every step, resolution is "twice" better)
        self.base_downgrade = math.floor(math.log2(min(base_resolution)))
        self.max_steps = max_steps
        self.sensing_cost = sensing_cost
        self.max_sensing_budget = max_sensing_budget
        #distance the agent will move when call a movement action
        self.move_magnitude = move_magnitude
        self.rotate_degrees = rotate_degrees
        # angle degree when looking up or down
        self.look_degrees = look_degrees
        # distance within an object can be seen
        self.visibility_distance = visibility_distance
        self.success_distance = success_distance
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)  

        self.processor = AutoProcessor.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name).to(self.device)
        self.encoder.eval()
        
        # controller
        self.controller = Controller(
            platform = CloudRendering,
            width = self.base_resolution[0],
            height = self.base_resolution[1],
            visibilityDistance = self.visibility_distance,
            renderDepthImage = False,
            renderInstanceSegmentation = False,
        )
        self.controller.start()

        # action params
        self.action_list = [
            "MoveAhead",
            "MoveRight",
            "MoveLeft",
            "MoveBack",
            "RotateRight",
            "RotateLeft",
            "LookUp",
            "LookDown",
            "SENSE",
            "DONE",
        ]

        self.action_params = {
            "MoveAhead": {"moveMagnitude": self.move_magnitude},
            "MoveRight": {"moveMagnitude": self.move_magnitude},
            "MoveLeft": {"moveMagnitude": self.move_magnitude},
            "MoveBack": {"moveMagnitude": self.move_magnitude},
            "RotateRight": {"degrees": self.rotate_degrees},
            "RotateLeft": {"degrees": self.rotate_degrees},
            "LookUp": {"degrees": self.look_degrees},
            "LookDown": {"degrees": self.look_degrees},
        }
        
        # Reward params
        self.step_penalty = step_penalty
        self.sense_penalty = sense_penalty
        self.fail_penalty = fail_penalty
        self.bump_penalty = bump_penalty
        self.distance_scale = distance_scale
        self.success_reward = success_reward

    def init_new_scene(self, scene):
        # agent params init
        self.step_count = 0
        self.prev_action = 0
        self.current_downgrade = self.base_downgrade
        self.remaining_sensing_budget = self.max_sensing_budget
        self.target_obj_id = None
        self.target_obj_type = None
        self.last_distance = None
        self.closest_distance = None
        self.action_history = []
        self.reward_history = []
        
        # scene and controller
        self.scene = scene
        self.controller.reset(scene)
        event = self.controller.step(
            action="Initialize",
            gridSize=self.move_magnitude,
            renderImage=True,
        )
        self.controller.step(action='InitialRandomSpawn', 
                             randomSeed=self.seed, 
                             forceVisible=True, 
                             numRepeats=1)
        
        self._define_target()
    
    def reset(self, target_obj_type):
        #to use at the end of an epoch (step?   )
        self.controller.reset(self.scene)

        #initialize
        event = self.controller.step(
            action="Initialize",
            gridSize=self.move_magnitude,
            visibilityDistance=self.visibility_distance,
            renderImage=True,
        )

        #reset episode state
        self.step_count = 0
        self.current_action = None
        self.current_downgrade = self.base_downgrade
        self.remaining_sensing_budget = self.max_sensing_budget
        self.last_distance = None
        self.closest_distance = None
        self.action_history = []

        #next/new   object cible
        if self.target_obj_type is None:
            self.target_obj_type = random.choice(self.target_object_types)
        else:
            self.target_obj_type = target_obj_type

        candidate_objects = [
            obj for obj in event.metadata["objects"]
            if obj["objectType"] == self.target_obj_type
        ]

        assert len(candidate_objects) == 0

        self.last_distance = self._get_distance_to_object(event)
        self.closest_distance = self.last_distance

        obs = self.get_obs(event)

        info = {
            "target_obj_type": self.target_obj_type,
            "num_candidates": len(self.candidate_objects),
        }

        return obs, info


    def get_obs(self, event=None):
        if event is None:
            event = self.current_event

        frame = event.frame

        inputs = self.processor(images=frame, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            vis_feat = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()

        agent_meta = event.metadata["agent"]

        pos = agent_meta["position"]
        rot = agent_meta["rotation"]

        gps = torch.tensor([pos["x"], pos["y"], pos["z"]], dtype=torch.float32)

        compass = torch.tensor([rot["y"] / 360.0, agent_meta.get("cameraHorizon", 0.0) / 360.0], dtype=torch.float32)

        #self.prev_action = 0 => prev_action = [1, 0, 0, ..]
        #self.prev_action = 1 => prev_action = [0, 1 0, ..]        
        prev_action = F.one_hot(torch.tensor(self.prev_action), num_classes=len(self.action_list)).float()

        res_level = torch.tensor([float(self.current_downgrade)], dtype=torch.float32)

        remaining_budget = torch.tensor([float(self.remaining_sensing_budget)], dtype=torch.float32)

        obs = torch.cat([vis_feat, gps, compass, prev_action, res_level, remaining_budget], dim=0)

        return obs
    
    #TODO
    
    def _define_target(self):
        event = self.controller.last_event
        objects = event.metadata["objects"]
        if self.target_object_types: valid_objects = [(i, obj) for i, obj in enumerate(objects) if (obj["pickupable"]) and (self._get_distance_to_object(obj["position"])>self.visibility_distance) and (obj['objectType'] in self.target_object_types)]
        else: valid_objects = [(i, obj) for i, obj in enumerate(objects) if obj["pickupable"] and self._get_distance_to_object(obj["position"])>self.visibility_distance]
        if len(valid_objects) == 0:
            raise ValueError("No valid objects found for target selection")
        i, obj = random.choice(valid_objects)
        self.target_obj_idx = i
        self.target_obj_pos = obj["position"]
        
    def _get_distance_to_object(self, obj_pos):
        agent_pos = self.current_event.metadata["agent"]["position"]

        dx = agent_pos["x"] - obj_pos["x"]
        dy = agent_pos["y"] - obj_pos["y"]
        dz = agent_pos["z"] - obj_pos["z"]

        return np.sqrt(dx*dx + dy*dy + dz*dz)

    def _compute_reward(self):
        #compute the reward of a step (step penalty, sensing penalty, progress toward the target)
        '''step penalty - distance reduction reward - sensing penalty - success reward or fail'''
        reward = 0.0
        
        diff_distance = (self.closest_distance - self._get_distance_to_object(self.target_obj_pos))
        if diff_distance > 0:
            reward += self.distance_scale*diff_distance
            self.closest_distance = self._get_distance_to_object(self.target_obj_pos)
            
        if self.current_action=='SENSE': reward -= self.sense_penalty
        elif self.current_action=='DONE' and self._check_success(): reward += self.success_reward
        elif self.current_action=='DONE' and not self._check_success(): reward -= self.fail_penalty
        elif not self.controller.last_event.metadata['lastActionSuccess']: reward -= self.bump_penalty
        else: reward = -self.step_penalty
        
        if self._end_checker(): reward -= self.fail_penalty
        self.reward_history.append(reward)
        return reward
    
    def _end_checker(self):
        '''Check if agent has reached sensing or time quota'''
        return ((self.remaining_sensing_budget<=0) or (self.step_count>=self.max_steps))
    
    def _check_success(self):
        return self.controller.last_event.metadata["objects"][self.target_obj_idx]["visible"] and (self._get_distance_to_object(self.target_obj_pos)<=self.success_distance)
    
    def learn(self):
        pass

    def step(self, action):
        #apply the action chosen, update internal state
        #gère navigation / sensing / stop
        #calcule done, reward, next_obs et info
        #renvoie la transition RL standard
        self.current_event = self.controller.last_event
        return next_obs, reward, done

    def close(self):
        try:
            self.controller.stop()
        except Exception:
            pass