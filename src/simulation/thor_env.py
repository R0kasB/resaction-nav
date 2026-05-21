import os
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

from src.utils.image_resolution import degrade_resolution

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

MINIMAL_ACTION_LIST = [
    "MoveAhead",
    "RotateLeft",
    "RotateRight",
    # "DONE",
]

NAVIGATION_ACTION_LIST = [
    action for action in ACTION_LIST if action != "SENSE"
]

ACTION_SETS = {
    "full": ACTION_LIST,
    "minimal": MINIMAL_ACTION_LIST,
    "navigation": NAVIGATION_ACTION_LIST,
}


class ThorEnv:
    """
    AI2-THOR navigation environment with dynamic resolution.

    Interface follows Gymnasium conventions:
      obs                            = env.reset(scene, target_obj_type)
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

    # ------------------------------------------------------------------
    # Aux-features layout constants
    # These are the *dimensionalities* of each block, in order.
    # The offset of any block = cumulative sum of previous blocks.
    # Centralising here means:
    #   - baselines can read env.budget_idx instead of hardcoding an offset,
    #   - PolicyLSTM can compute input_dim from env.aux_features_dim,
    #   - changing the layout auto-propagates to all consumers.
    # ------------------------------------------------------------------
    _AUX_GPS_DIM        = 2   # x, z  (y dropped: agent height is ~constant)
    _AUX_COMPASS_DIM    = 4   # sin(yaw), cos(yaw), sin(horizon), cos(horizon)
    _AUX_RESOLUTION_DIM = 1
    _AUX_BUDGET_DIM     = 1
    _AUX_TARGET_IDX_DIM = 1   # we emit the idx, not the embedding

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
        fixed_high_res: bool = False,
        action_set: str = "full",
        auto_success_on_goal: bool = False,
        env_id: int = 0,
        randomize_object_spawn: bool = True,
    ):
        self.target_object_types = target_object_types
        self.base_resolution = base_resolution
        self.success_distance = success_distance
        self.base_downgrade = math.floor(math.log2(min(base_resolution)))
        self.max_steps = max_steps
        self.max_sensing_budget = max_sensing_budget
        self.move_magnitude = move_magnitude
        self.rotate_degrees = rotate_degrees
        self.look_degrees = look_degrees
        self.visibility_distance = visibility_distance
        self.target_object_embed_dim = target_object_embed_dim

        # Closed-set target vocabulary. Populated by set_target_vocab() from
        # train.py, sourced from training.target.candidates in the YAML config.
        # Kept in the env (not just the policy) because get_aux_features() must
        # emit the integer index that the policy's nn.Embedding will look up.
        self._target_vocab: list[str] = []
        self._target_to_idx: dict[str, int] = {}

        self.seed = seed
        self.cfg = reward_cfg or RewardConfig()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        if action_set not in ACTION_SETS:
            raise ValueError(
                f"Unknown action_set={action_set!r}. "
                f"Choose from {list(ACTION_SETS.keys())}."
            )

        self.fixed_high_res = fixed_high_res
        self.action_set = action_set
        self.action_list = list(ACTION_SETS[action_set])
        self.auto_success_on_goal = auto_success_on_goal

        if "DONE" not in self.action_list and not self.auto_success_on_goal:
            raise ValueError(
                "This action_set does not contain DONE. "
                "Set auto_success_on_goal=True or use an action_set with DONE."
            )

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

        controller_options = {
            "width": base_resolution[0],
            "height": base_resolution[1],
            "visibilityDistance": visibility_distance,
            "renderDepthImage": False,
            "renderInstanceSegmentation": False,
            "platform": CloudRendering,
            "gpu_device": 0,
        }
        if controller_kwargs:
            controller_options.update(controller_kwargs)

        self.controller = Controller(**controller_options)

        self.env_id = env_id
        self._reset_count = 0
        self.randomize_object_spawn = randomize_object_spawn

        # episode state — populated by reset()
        self._step_count = 0
        self._current_action = "MoveAhead"
        self._current_downgrade = 0 if self.fixed_high_res else self.base_downgrade
        self._remaining_sensing_budget = 0 if self.fixed_high_res else self.max_sensing_budget
        self._last_sense_was_valid = True
        self._closest_distance = np.inf
        self._done = False
        self.current_event = None
        self.target_obj_type = None
        self.scene = None
        self._scene_bounds = None
        self._agent_start = None

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    def set_target_vocab(self, vocab: list[str]) -> None:
        """Register the closed list of object names the policy can embed.

        Must be called once after env construction and before reset(). Ordering
        is preserved verbatim — train.py is responsible for sorting it for
        deterministic idx assignment.
        """
        self._target_vocab = list(vocab)
        self._target_to_idx = {name: i for i, name in enumerate(self._target_vocab)}

    # ------------------------------------------------------------------
    # Aux-features layout properties
    # ------------------------------------------------------------------

    @property
    def gps_idx(self) -> int:
        return 0

    @property
    def compass_idx(self) -> int:
        return self._AUX_GPS_DIM

    @property
    def prev_action_idx_offset(self) -> int:
        return self._AUX_GPS_DIM + self._AUX_COMPASS_DIM

    @property
    def resolution_idx(self) -> int:
        return self.prev_action_idx_offset + len(self.action_list)

    @property
    def budget_idx(self) -> int:
        """Position of the sensing-budget scalar inside get_aux_features() output."""
        return self.resolution_idx + self._AUX_RESOLUTION_DIM

    @property
    def target_idx_position(self) -> int:
        """Position of the target-object index (last scalar)."""
        return self.budget_idx + self._AUX_BUDGET_DIM

    @property
    def aux_features_dim(self) -> int:
        """Total dimensionality of the vector returned by get_aux_features()."""
        return (
            self._AUX_GPS_DIM
            + self._AUX_COMPASS_DIM
            + len(self.action_list)
            + self._AUX_RESOLUTION_DIM
            + self._AUX_BUDGET_DIM
            + self._AUX_TARGET_IDX_DIM
        )

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

        if self.randomize_object_spawn:
            spawn_seed = int(self.seed) + 100_000 * int(self.env_id) + int(self._reset_count)
        else:
            spawn_seed = int(self.seed)

        self.current_event = self.controller.step(
            action="InitialRandomSpawn",
            randomSeed=spawn_seed,
            forceVisible=True,
        )

        # Cache scene bounds and agent start for GPS normalisation in get_aux_features().
        event = self.controller.step("GetReachablePositions")
        positions = event.metadata["actionReturn"]
        xs = [p["x"] for p in positions]
        zs = [p["z"] for p in positions]
        self._scene_bounds = {
            "x_min": min(xs), "x_max": max(xs),
            "z_min": min(zs), "z_max": max(zs),
        }
        agent_position = self.current_event.metadata["agent"]["position"]
        self._agent_start = (agent_position["x"], agent_position["z"])

        self._reset_count += 1

        self._step_count = 0
        self._current_action = "MoveAhead"
        self._current_downgrade = 0 if self.fixed_high_res else self.base_downgrade
        self._remaining_sensing_budget = 0 if self.fixed_high_res else self.max_sensing_budget

        if self.fixed_high_res and self._current_downgrade != 0:
            raise RuntimeError(
                f"fixed_high_res=True but reset set _current_downgrade={self._current_downgrade}"
            )
        self._last_sense_was_valid = True
        self._done = False

        if target_obj_type:
            self.target_obj_type = target_obj_type
        else:
            self._define_target()

        self._closest_distance = self._get_min_distance_to_object(self.target_obj_type)
        self._initial_distance = self._closest_distance
        self._min_distance_seen = self._closest_distance

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

        if action == "SENSE":
            self._last_sense_was_valid = (
                self._current_downgrade > 0 and self._remaining_sensing_budget > 0
            )
            if self._last_sense_was_valid:
                self._current_downgrade -= 1
                self._remaining_sensing_budget -= 1

        if action in MOVE_ACTIONS and not self.fixed_high_res:
            self._current_downgrade = self.base_downgrade

        truncated = self._fail_checker()
        obs = self._compute_obs()

        current_distance = self._get_min_distance_to_object(self.target_obj_type)
        self._min_distance_seen = min(self._min_distance_seen, current_distance)

        task_success = self._check_success()
        reward = self._compute_reward(truncated=truncated, task_success=task_success)

        terminated = action == "DONE" or (
            self.auto_success_on_goal and task_success
        )

        if self.fixed_high_res and self._current_downgrade != 0:
            raise RuntimeError(
                f"fixed_high_res=True but _current_downgrade={self._current_downgrade} "
                f"after action={action}, step={self._step_count}"
            )

        info = {
            "step": self._step_count,
            "downgrade": self._current_downgrade,
            "sensing_budget": self._remaining_sensing_budget,
            "success": task_success,
            "task_success": task_success,
            "auto_success_on_goal": self.auto_success_on_goal,
            "initial_distance": self._initial_distance,
            "current_distance": current_distance,
            "min_distance_seen": self._min_distance_seen,
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

    def _get_target_object_idx(self) -> int:
        """Return the integer index of the current target object.

        The policy's nn.Embedding maps this idx to a learned vector. We emit the
        raw idx (not the embedding) so the embedding stays a learnable
        parameter in the policy, rather than a fixed function of the name.
        """
        if not self._target_to_idx:
            raise RuntimeError(
                "Target vocabulary not set. Call env.set_target_vocab([...]) "
                "before reset()."
            )
        name = self.target_obj_type or ""
        if name not in self._target_to_idx:
            raise KeyError(
                f"Target object '{name}' is not in the configured vocabulary "
                f"{self._target_vocab}. Add it to training.target.candidates "
                "in the YAML config."
            )
        return self._target_to_idx[name]

    # ------------------------------------------------------------------
    # Reward / termination
    # ------------------------------------------------------------------

    def _compute_reward(self, truncated: bool, task_success: bool | None = None) -> float:
        if task_success is None:
            task_success = self._check_success()
        reward = 0.0
        action = self._current_action
        cfg = self.cfg

        current_dist = self._get_min_distance_to_object(self.target_obj_type)
        progress = self._closest_distance - current_dist
        if progress > 0:
            reward += cfg.distance_scale * progress
            self._closest_distance = current_dist

        if action == "SENSE":
            reward -= cfg.sense_penalty if self._last_sense_was_valid else cfg.oversensing_penalty
        elif action == "DONE":
            reward += cfg.success_reward if task_success else -cfg.fail_penalty
        elif not self.current_event.metadata["lastActionSuccess"]:
            reward -= cfg.bump_penalty
        else:
            reward -= cfg.step_penalty

        if self.auto_success_on_goal and task_success and action != "DONE":
            reward += cfg.success_reward

        if truncated and not task_success:
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

    # ------------------------------------------------------------------
    # Aux features
    # ------------------------------------------------------------------

    def get_aux_features(self, prev_action_idx: int | None = None) -> torch.Tensor:
        """
        Build non-visual features for the policy.

        Layout (see the *_idx properties for byte-exact offsets):
          - gps                : 2  (x, z, scene-normalised to [-1, 1])
          - compass            : 4  (sin/cos of yaw and horizon)
          - prev_action one-hot: n_actions
          - resolution level   : 1  (current_downgrade / base_downgrade)
          - sensing budget     : 1  (remaining / max)
          - target object idx  : 1  (cast to float; policy casts back to long
                                     for nn.Embedding lookup)
        Total dim = self.aux_features_dim.
        """
        if self.current_event is None:
            raise RuntimeError("Environment has not been reset yet.")

        metadata = self.current_event.metadata
        agent_metadata = metadata["agent"]
        position = agent_metadata["position"]
        rotation = agent_metadata["rotation"]

        # GPS: normalise (x, z) to [-1, 1] using the scene's reachable bounds.
        # y (agent height) is ~constant in AI2-THOR and carries no navigation info.
        b = self._scene_bounds
        x_norm = 2 * (position["x"] - b["x_min"]) / (b["x_max"] - b["x_min"] + 1e-6) - 1
        z_norm = 2 * (position["z"] - b["z_min"]) / (b["z_max"] - b["z_min"] + 1e-6) - 1
        gps = torch.tensor([x_norm, z_norm], dtype=torch.float32, device=self.device)

        # Compass: sin/cos encoding avoids wrap-around discontinuity at 0°/360°.
        yaw_rad = math.radians(rotation["y"])
        horizon_rad = math.radians(agent_metadata.get("cameraHorizon", 0.0))
        compass = torch.tensor(
            [
                math.sin(yaw_rad), math.cos(yaw_rad),
                math.sin(horizon_rad), math.cos(horizon_rad),
            ],
            dtype=torch.float32, device=self.device,
        )

        # Previous action one-hot.
        prev_action = torch.zeros(
            len(self.action_list), dtype=torch.float32, device=self.device,
        )
        if prev_action_idx is not None:
            prev_action[prev_action_idx] = 1.0

        # Resolution level, normalised to [0, 1].
        resolution_level = torch.tensor(
            [self._current_downgrade / max(self.base_downgrade, 1)],
            dtype=torch.float32, device=self.device,
        )

        # Sensing budget, normalised to [0, 1].
        sensing_budget = torch.tensor(
            [self._remaining_sensing_budget / max(self.max_sensing_budget, 1)],
            dtype=torch.float32, device=self.device,
        )

        # Target-object index (NOT the embedding).
        # Cast to float for homogeneous cat; policy casts back to long for nn.Embedding.
        target_idx = torch.tensor(
            [float(self._get_target_object_idx())],
            dtype=torch.float32, device=self.device,
        )

        return torch.cat(
            [gps, compass, prev_action, resolution_level, sensing_budget, target_idx],
            dim=0,
        )