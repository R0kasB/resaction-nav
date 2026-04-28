import image_resolution
import torch
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
        scene,
        target_object_types,
        base_resolution = (224, 224),
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
    ):
        self.scene = scene
        self.target_object_types = target_object_types
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

        self.controller = Controller(
            scene=self.scene,
            platform=CloudRendering,
            width = self.base_resolution[0],
            height = self.base_resolution[1],
            visibilityDistance = self.visibility_distance,
            renderDepthImage=False,
            renderInstanceSegmentation=False,
        )

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

        self.step_count = 0
        self.prev_action = 0
        self.current_downgrade = self.base_downgrade
        self.remaining_sensing_budget = self.max_sensing_budget
        self.target_obj_id = None
        self.target_obj_type = None
        self.last_distance = None
        self.action_history = []   

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
        self.prev_action = 0
        self.current_downgrade = self.base_downgrade
        self.remaining_sensing_budget = self.max_sensing_budget
        self.last_distance = None
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

        self.last_distance = min(
            obj.get("distance", float("inf")) for obj in self.candidate_objects
        )

        obs = self.get_obs(event)

        info = {
            "target_obj_type": self.target_obj_type,
            "num_candidates": len(self.candidate_objects),
        }

        return obs, info


    def get_obs(self, event=None):
        if event is None:
            event = self.controller.last_event

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
    def _get_distance_to_target(self, ...):
        #compute the minimum distance to the categorie cible
        pass

    def _check_success(self, ...):
        #check qu'une isntance de la catégorie cible soit suffisament proche et visible
        #et check que la condition finale du succès (cible doit être au milieu de la cible, ...
        pass

    def _compute_reward(self, ...):
        #compute the reward of a step (step penalty, sensing penalty, progress toward the target)
        pass

    def step(self, action_idx):
        #apply the action chosen, update internal state
        #gère navigation / sensing / stop
        #calcule done, reward, next_obs et info
        #renvoie la transition RL standard
        pass

    def close(self):
        try:
            self.controller.stop()
        except Exception:
            pass