import os
import math
import random
from hashlib import sha256
from dataclasses import dataclass

import numpy as np
import torch
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

from src.utils.image_resolution import degrade_resolution

# https://allenai.github.io/ai2thor-v2.1.0-documentation/actions/initialization
# https://gymnasium.farama.org/introduction/basic_usage/


@dataclass
class RewardConfig:
    step_penalty: float = 0.002
    sense_penalty: float = 0.02
    oversensing_penalty: float = 0.05
    bump_penalty: float = 0.03
    fail_penalty: float = 1.0
    success_reward: float = 5.0
    distance_scale: float = 0.01


MOVE_ACTIONS = {"MoveAhead", "MoveRight", "MoveLeft", "MoveBack"}

ACTION_LIST = [
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


class ThorEnv:
    """
    AI2-THOR navigation environment with dynamic resolution.

    Interface follows Gymnasium conventions:
      obs                          = env.reset(scene, target_obj_type)
      obs, reward, term, trunc, info = env.step(action_idx)

    Observations are raw RGB frames as (3, H, W) float tensors in [0, 1].
    The LSTM policy in src/models/lstm.py is expected to encode them further.

    Actions (indexed 0-9):
      0-3: Move (Ahead/Right/Left/Back)  — resets resolution to base_downgrade
      4-5: Rotate (Right/Left)
      6-7: Look (Up/Down)
      8:   SENSE — halves current_downgrade if budget remains
      9:   DONE  — ends episode

    Resolution downgrade:
      0 = full resolution
      k = blocks of 2^k pixels (larger k = lower resolution)
      base_downgrade = floor(log2(min(resolution))) = worst/lowest level.
      SENSE halves the downgrade level (better resolution).
      Moving resets it back to base_downgrade (worst resolution).
    """

    def __init__(
        self,
        target_object_types=None,
        base_resolution=(224, 224),
        success_distance=0.5,
        max_steps=200,
        max_sensing_budget=5,
        move_magnitude=0.25,
        rotate_degrees=45.0,
        look_degrees=15.0,
        visibility_distance=1.5,
        target_object_embed_dim=8,
        controller_kwargs=None,
        device=None,
        seed=0,
        reward_cfg: RewardConfig = None,
    ):
        self.target_object_types = target_object_types
        self.base_resolution = base_resolution
        self.success_distance = success_distance
        # largest downgrade level = lowest resolution (e.g. floor(log2(224)) = 7)
        self.base_downgrade = math.floor(math.log2(min(base_resolution)))
        self.max_steps = max_steps
        self.max_sensing_budget = max_sensing_budget
        self.move_magnitude = move_magnitude
        self.rotate_degrees = rotate_degrees
        self.look_degrees = look_degrees
        self.visibility_distance = visibility_distance
        self.target_object_embed_dim = target_object_embed_dim
        self.seed = seed
        self.cfg = reward_cfg or RewardConfig()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.action_list = ACTION_LIST
        self.action2id = {a: i for i, a in enumerate(self.action_list)}
        self.action_params = {
            "MoveAhead":   {"moveMagnitude": move_magnitude},
            "MoveRight":   {"moveMagnitude": move_magnitude},
            "MoveLeft":    {"moveMagnitude": move_magnitude},
            "MoveBack":    {"moveMagnitude": move_magnitude},
            "RotateRight": {"degrees": rotate_degrees},
            "RotateLeft":  {"degrees": rotate_degrees},
            "LookUp":      {"degrees": look_degrees},
            "LookDown":    {"degrees": look_degrees},
        }

        #ryan
        controller_options = {
            "width": base_resolution[0],
            "height": base_resolution[1],
            "visibilityDistance": visibility_distance,
            "renderDepthImage": False,
            "renderInstanceSegmentation": False,
            "platform": CloudRendering,  #line from old
            "gpu_device": int(os.environ.get("CUDA_VISIBLE_DEVICES", 0)), #line from old as we will train on cluster
        }
        if controller_kwargs:
            controller_options.update(controller_kwargs)

        self.controller = Controller(**controller_options)
        
        #older
        # self.controller = Controller(
        #     width=base_resolution[0],
        #     height=base_resolution[1],
        #     visibilityDistance=visibility_distance,
        #     renderDepthImage=False,
        #     renderInstanceSegmentation=False,
        #     platform=CloudRendering,
        #     gpu_device=0,
        # )

        # episode state — populated by reset()
        self._step_count = 0
        self._current_action = "MoveAhead"
        self._current_downgrade = self.base_downgrade
        self._remaining_sensing_budget = max_sensing_budget
        self._last_sense_was_valid = True
        self._closest_distance = np.inf
        self._done = False
        self.current_event = None
        self.target_obj_type = None
        self.scene = None

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, scene: str, target_obj_type: str = None) -> torch.Tensor:
        """Start a new episode. Returns the first observation."""
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

        self._step_count = 0
        self._current_action = "MoveAhead"
        self._current_downgrade = self.base_downgrade
        self._remaining_sensing_budget = self.max_sensing_budget
        self._last_sense_was_valid = True
        self._done = False

        if target_obj_type:
            self.target_obj_type = target_obj_type
        else:
            self._define_target()

        self._closest_distance = self._get_min_distance_to_object(self.target_obj_type)
        return self._compute_obs()

    def step(self, action_idx: int):
        """
        Execute one action.
        Returns: (obs, reward, terminated, truncated, info)
        """
        action = self.action_list[action_idx]
        self._step_count += 1
        self._current_action = action

        if action in self.action_params:
            self.current_event = self.controller.step(
                action=action, **self.action_params[action]
            )

        # Set validity flag before reward (reward checks it).
        if action == "SENSE":
            self._last_sense_was_valid = (
                self._current_downgrade > 0 and self._remaining_sensing_budget > 0
            )
            if self._last_sense_was_valid:
                self._current_downgrade -= 1
                self._remaining_sensing_budget -= 1

        # Moving immediately resets resolution — new position = new view at worst res.
        if action in MOVE_ACTIONS:
            self._current_downgrade = self.base_downgrade

        truncated = self._fail_checker()
        obs = self._compute_obs()
        reward = self._compute_reward(truncated)
        terminated = action == "DONE"
        info = {
            "step": self._step_count,
            "downgrade": self._current_downgrade,
            "sensing_budget": self._remaining_sensing_budget,
            "success": self._check_success() if terminated else False,
        }

        self._done = terminated or truncated
        return obs, reward, terminated, truncated, info

    def close(self):
        try:
            self.controller.stop()
        except Exception:
            pass

    def update_seed(self, seed: int):
        self.seed = seed

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _compute_obs(self) -> torch.Tensor:
        if self.current_event is None or getattr(self.current_event, "frame", None) is None:
            raise RuntimeError("No current frame available. Call reset() before requesting observations.")

        frame = np.ascontiguousarray(self.current_event.frame)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected RGB frame with shape (H, W, 3), got {frame.shape!r}")

        try:
            tensor = torch.from_numpy(frame).permute(2, 0, 1).to(dtype=torch.float32)
        except RuntimeError:
            # Some Torch builds are missing NumPy bridge support at runtime.
            tensor = torch.tensor(frame.tolist(), dtype=torch.float32).permute(2, 0, 1)
        tensor = tensor / 255.0
        max_level = int(math.floor(math.log2(min(tensor.shape[1], tensor.shape[2]))))
        level = int(max(0, min(self._current_downgrade, max_level)))
        return degrade_resolution(tensor, level)

    # ------------------------------------------------------------------
    # Target / distance helpers
    # ------------------------------------------------------------------

    def _define_target(self):
        objects = self.current_event.metadata["objects"]
        valid = [
            o for o in objects
            if o["pickupable"]
            and (self.target_object_types is None or o["objectType"] in self.target_object_types)
        ]
        if not valid:
            raise ValueError(f"No valid target objects found in scene '{self.scene}'")
        self.target_obj_type = random.choice(valid)["objectType"]

    def _get_min_distance_to_object(self, obj_type: str) -> float:
        agent_pos = self.current_event.metadata["agent"]["position"]
        agent_vec = np.array([agent_pos["x"], agent_pos["y"], agent_pos["z"]], dtype=np.float32)
        positions = np.array([
            [o["position"]["x"], o["position"]["y"], o["position"]["z"]]
            for o in self.current_event.metadata["objects"]
            if o["objectType"] == obj_type
        ], dtype=np.float32)
        if len(positions) == 0:
            return np.inf
        return float(np.linalg.norm(positions - agent_vec, axis=1).min())

    def _get_distance_to_position(self, obj_pos: dict) -> float:
        agent_pos = self.current_event.metadata["agent"]["position"]
        agent_vec = np.array([agent_pos["x"], agent_pos["y"], agent_pos["z"]], dtype=np.float32)
        obj_vec = np.array([obj_pos["x"], obj_pos["y"], obj_pos["z"]], dtype=np.float32)
        return float(np.linalg.norm(agent_vec - obj_vec))

    def _get_target_object_embedding(self) -> torch.Tensor:
        embed_dim = int(getattr(self, "target_object_embed_dim", 0))
        if embed_dim <= 0:
            return torch.empty(0, dtype=torch.float32, device=self.device)

        target_obj_type = self.target_obj_type or ""
        digest = sha256(target_obj_type.encode("utf-8")).digest()
        values = [
            ((digest[i % len(digest)] / 255.0) * 2.0) - 1.0
            for i in range(embed_dim)
        ]
        return torch.tensor(values, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # Reward / termination
    # ------------------------------------------------------------------

    def _compute_reward(self, truncated: bool) -> float:
        reward = 0.0
        action = self._current_action
        cfg = self.cfg

        current_dist = self._get_min_distance_to_object(self.target_obj_type)
        progress = self._closest_distance - current_dist
        if progress > 0:
            reward += cfg.distance_scale * progress
            self._closest_distance = current_dist

        if action == "SENSE":
            reward -= cfg.sense_penalty
            if not self._last_sense_was_valid:
                reward -= cfg.oversensing_penalty
        elif action == "DONE":
            reward += cfg.success_reward if self._check_success() else -cfg.fail_penalty
        elif not self.current_event.metadata["lastActionSuccess"]:
            reward -= cfg.bump_penalty
        else:
            reward -= cfg.step_penalty

        if truncated:
            reward -= cfg.fail_penalty

        return reward

    def _fail_checker(self) -> bool:
        return self._step_count >= self.max_steps

    def _check_success(self) -> bool:
        targets = [
            o for o in self.current_event.metadata["objects"]
            if o["objectType"] == self.target_obj_type
        ]
        return any(
            o["visible"] and self._get_distance_to_position(o["position"]) <= self.success_distance
            for o in targets
        )

    def get_aux_features(self, prev_action_idx: int | None = None) -> torch.Tensor:
        """
        Build non-visual features for the policy.

        Returns:
            Tensor with shape (17 + target_object_embed_dim,):
            - GPS position: 3
            - compass: 2
            - previous action one-hot: n_actions
            - resolution level: 1
            - remaining sensing budget: 1
            - target object embedding: target_object_embed_dim
        """
        if self.current_event is None:
            raise RuntimeError("Environment has not been reset yet.")

        metadata = self.current_event.metadata
        agent_metadata = metadata["agent"]

        position = agent_metadata["position"]
        rotation = agent_metadata["rotation"]

        gps = torch.tensor(
            [
                position["x"],
                position["y"],
                position["z"],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        compass = torch.tensor(
            [
                rotation["y"] / 360.0,
                agent_metadata.get("cameraHorizon", 0.0) / 360.0,
            ],
            dtype=torch.float32,
            device=self.device,
        )

        prev_action = torch.zeros(
            len(self.action_list),
            dtype=torch.float32,
            device=self.device,
        )

        if prev_action_idx is not None:
            prev_action[prev_action_idx] = 1.0

        resolution_level = torch.tensor(
            [
                self._current_downgrade / max(self.base_downgrade, 1),
            ],
            dtype=torch.float32,
            device=self.device,
        )

        sensing_budget = torch.tensor(
            [
                self._remaining_sensing_budget / max(self.max_sensing_budget, 1),
            ],
            dtype=torch.float32,
            device=self.device,
        )
        target_object_embedding = self._get_target_object_embedding()

        return torch.cat(
            [
                gps,
                compass,
                prev_action,
                resolution_level,
                sensing_budget,
                target_object_embedding,
            ],
            dim=0,
        )
