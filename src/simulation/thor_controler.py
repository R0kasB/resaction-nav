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
        encoder_name: str = "facebook/dinov2-base",
        device = 'cuda',
        seed = 0,
        step_penalty = 0.002,
        distance_scale = 0.01,
        sense_penalty = 0.02,
        oversensing_penalty = 0.05,
        bump_penalty = 0.03,
        fail_penalty = 1.0,
        success_reward = 5.0,
    ):
        self.target_object_types = target_object_types
        self.success_distance = success_distance
        self.base_resolution = base_resolution
        #start with the lowest resolution (every step, resolution is "twice" better)
        self.base_downgrade = math.floor(math.log2(min(base_resolution))) #TODO
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

        #self.processor = AutoProcessor.from_pretrained(encoder_name)
        #self.encoder = AutoModel.from_pretrained(encoder_name).to(self.device)
        #self.encoder.eval()
        
        # controller
        self.controller = Controller(
            #platform = CloudRendering,
            width = self.base_resolution[0],
            height = self.base_resolution[1],
            visibilityDistance = self.visibility_distance,
            renderDepthImage = False,
            renderInstanceSegmentation = False,
        )

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
        
        self.action2id = {a:i for i,a in enumerate(self.action_list)}
        
        # Reward params
        self.step_penalty = step_penalty
        self.sense_penalty = sense_penalty
        self.fail_penalty = fail_penalty
        self.bump_penalty = bump_penalty
        self.oversensing_penalty = oversensing_penalty
        self.distance_scale = distance_scale
        self.success_reward = success_reward
    
    def init_new_scene(self, scene, target_obj_type = None):
        self.scene = scene
        self.controller.reset(scene)

        self.current_event = self.controller.step(
            action="Initialize",
            gridSize=self.move_magnitude,
            renderImage=True,
        )

        self.current_event = self.controller.step(
            action="InitialRandomSpawn",
            randomSeed=self.seed,
            forceVisible=True,
        )

        self.step_count = 0
        self.current_action = "MoveAhead"
        self.current_downgrade = self.base_downgrade
        self.remaining_sensing_budget = self.max_sensing_budget
        self.closest_distance = np.inf
        self.end_status = False
        self.current_reward = 0
        self.trajectory = []

        if target_obj_type: self.target_obj_type = target_obj_type 
        else: self._define_target()
        self.closest_distance = self._get_min_distance_to_object(self.target_obj_type)

    '''def _compute_obs(self):
        
        frame = self.current_event.frame

        inputs = self.processor(images=frame, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            vis_feat = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()

        agent_meta = self.current_event.metadata["agent"]

        pos = agent_meta["position"]
        rot = agent_meta["rotation"]

        gps = torch.tensor([pos["x"], pos["y"], pos["z"]], dtype=torch.float32)
        compass = torch.tensor([rot["y"] / 360.0, agent_meta.get("cameraHorizon", 0.0) / 360.0], dtype=torch.float32)

        #self.current_action = 0 => current_action = [1, 0, 0, ..]
        #self.current_action = 1 => current_action = [0, 1 0, ..]
        action_id = self.action2id[self.current_action]
        current_action = F.one_hot(torch.tensor(action_id), num_classes=len(self.action_list)).float()
        res_level = torch.tensor([float(self.current_downgrade)], dtype=torch.float32)
        remaining_budget = torch.tensor([float(self.remaining_sensing_budget)], dtype=torch.float32)
        obs = torch.cat([vis_feat, gps, compass, current_action, res_level, remaining_budget], dim=0)
        return obs'''
        
    def _compute_obs(self):
        return None

    def _define_target(self):
        objects = self.current_event.metadata["objects"]
        valid_objects = [
            obj for obj in objects
            if obj["pickupable"]
            and (self.target_object_types is None or obj["objectType"] in self.target_object_types)
        ]
        if len(valid_objects) == 0:
            raise ValueError("No valid objects found")
        
        obj = random.choice(valid_objects)
        self.target_obj_type = obj["objectType"]
        
    def _get_min_distance_to_object(self, obj_type):
        agent_pos = self.current_event.metadata["agent"]["position"]
        
        target_positions = np.array([
            [obj["position"]["x"], obj["position"]["y"], obj["position"]["z"]]
            for obj in self.current_event.metadata["objects"]
            if obj["objectType"] == obj_type
        ], dtype=np.float32)
        agent_vector = np.array(
            [agent_pos["x"], agent_pos["y"], agent_pos["z"]],
            dtype=np.float32
        )
        
        distances = np.linalg.norm(target_positions - agent_vector, axis=1)
        return distances.min()
    
    def _get_distance_to_position(self, obj_pos):
        agent_pos = self.current_event.metadata["agent"]["position"]
        agent_vec = np.array(
            [agent_pos["x"], agent_pos["y"], agent_pos["z"]],
            dtype=np.float32
        )
        obj_vec = np.array(
            [obj_pos["x"], obj_pos["y"], obj_pos["z"]],
            dtype=np.float32
        )
        return np.linalg.norm(agent_vec - obj_vec)

    def _compute_reward(self):
        #compute the reward of a step (step penalty, sensing penalty, progress toward the target)
        '''step penalty - distance reduction reward - sensing penalty - success reward or fail'''
        reward = 0.0
        
        current_dist = self._get_min_distance_to_object(self.target_obj_type)
        diff_distance = (self.closest_distance - current_dist)
        if diff_distance > 0:
            reward += self.distance_scale*diff_distance
            self.closest_distance = current_dist
            
        if self.current_action=='SENSE' and self.current_downgrade > 2 and self.remaining_sensing_budget<=0: reward -= (self.sense_penalty + self.oversensing_penalty)
        elif self.current_action=='SENSE': reward -= self.sense_penalty
        elif self.current_action=='DONE' and self._check_success(): reward += self.success_reward
        elif self.current_action=='DONE' and not self._check_success(): reward -= self.fail_penalty
        elif not self.current_event.metadata['lastActionSuccess']: reward -= self.bump_penalty
        else: reward -= self.step_penalty
        
        if self._fail_checker(): 
            reward -= self.fail_penalty
        
        return reward
    
    def _fail_checker(self):
        '''Check if agent has reached sensing or time quota'''
        return self.step_count>=self.max_steps
    
    def _check_success(self):
        target_objs = [
            obj for obj in self.current_event.metadata["objects"]
            if obj["objectType"] == self.target_obj_type
        ]

        return any(
            obj["visible"]
            and self._get_distance_to_position(obj["position"]) <= self.success_distance
            for obj in target_objs
        )
        
    def update_seed(self, seed):
        self.seed = seed
    
    def run_scene(self, model):
        previous_obs = self._compute_obs()
        while not self.end_status:
            action_id = model(previous_obs)
            action = self.action_list[action_id]
            obs, reward, done = self.step(action)
            self.trajectory.append({
                "obs": previous_obs,
                "action": action,
                "reward": reward,
                "next_obs": obs,
                "done": done
            })
            previous_obs = obs
        return self.trajectory

    def step(self, action):
        self.step_count += 1

        if action not in ["DONE", "SENSE"]:
            self.current_event = self.controller.step(
                action=action,
                **self.action_params.get(action, {})
            )
        self.current_action = action

        obs = self._compute_obs()
        reward = self._compute_reward()
        done = (
            action == "DONE"
            or self._fail_checker()
        )

        self.current_reward = reward
        self.current_obs = obs
        self.end_status = done

        if action == "SENSE" and self.current_downgrade >= 2 and self.remaining_sensing_budget > 0:
            self.current_downgrade /= 2
            self.remaining_sensing_budget -= 1

        return obs, reward, done

    def close(self):
        try:
            self.controller.stop()
        except Exception:
            pass