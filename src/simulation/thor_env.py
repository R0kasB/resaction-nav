import os
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from ai2thor.util.metrics import get_shortest_path_to_object, path_distance

from src.utils.image_resolution import degrade_resolution


@dataclass
class RewardConfig:
    step_penalty: float = 0.002
    sense_penalty: float = 0.02
    oversensing_penalty: float = 0.05
    bump_penalty: float = 0.03
    fail_penalty: float = 1.0
    wrong_done_penalty: float = 1.5
    timeout_penalty: float = 0.5
    success_reward: float = 5.0
    distance_scale: float = 0.01
    success_distance: float = 1.5
    # standard potential-based shaping which keeps
    # the optimal policy invariant (Ng et al. 1999) when applied symmetrically.
    use_signed_progress: bool = True
    first_visibility_bonus: float = 0.5


MOVE_ACTIONS = {"MoveAhead", "MoveRight", "MoveLeft", "MoveBack"}

ACTION_LIST = [
    "MoveAhead",
    "MoveRight",
    "MoveLeft",
    "MoveBack",
    "RotateRight",
    "RotateLeft",
    #"LookUp",
    #"LookDown",
    "DONE",
    "SENSE",
]

MINIMAL_ACTION_LIST = [
    "MoveAhead",
    "RotateLeft",
    "RotateRight",
    "DONE",
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

    _AUX_GPS_DIM        = 2
    _AUX_COMPASS_DIM    = 4
    _AUX_RESOLUTION_DIM = 1
    _AUX_BUDGET_DIM     = 1
    _AUX_TARGET_IDX_DIM = 1

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
        start_downgrade: int | None = None,
    ):
        self.target_object_types = target_object_types
        self.base_resolution = base_resolution
        self.success_distance = success_distance
        self.base_downgrade = math.floor(math.log2(min(base_resolution)))
        self.start_downgrade = start_downgrade if start_downgrade is not None else self.base_downgrade
        self.max_steps = max_steps
        self.max_sensing_budget = max_sensing_budget
        self.move_magnitude = move_magnitude
        self.rotate_degrees = rotate_degrees
        self.look_degrees = look_degrees
        self.visibility_distance = visibility_distance
        self.target_object_embed_dim = target_object_embed_dim
        
        self.enforce_initial_distance = True

        self._target_vocab: list[str] = []
        self._target_to_idx: dict[str, int] = {}

        self.seed = seed
        self.cfg = reward_cfg or RewardConfig()

        self._geo_cache: dict[tuple, float] = {}

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

        # Episode state — populated by reset()
        self._step_count = 0
        self._current_action = "MoveAhead"
        self._current_downgrade = 0 if self.fixed_high_res else self.start_downgrade
        self._remaining_sensing_budget = 0 if self.fixed_high_res else self.max_sensing_budget
        self._last_sense_was_valid = True
        self._closest_distance = np.inf
        self._prev_distance = np.inf            
        self._target_ever_visible = False      
        self._optimal_path_length = float("inf")  
        self._path_length = 0.0                  
        self._prev_agent_xz = None               
        self._done = False
        self.current_event = None
        self.target_obj_type = None
        self.scene = None
        self._scene_bounds = None
        self._agent_start = None


    def set_target_vocab(self, vocab: list[str]) -> None:
        self._target_vocab = list(vocab)
        self._target_to_idx = {name: i for i, name in enumerate(self._target_vocab)}

    @property
    def gps_idx(self) -> int: return 0

    @property
    def compass_idx(self) -> int: return self._AUX_GPS_DIM

    @property
    def prev_action_idx_offset(self) -> int:
        return self._AUX_GPS_DIM + self._AUX_COMPASS_DIM

    @property
    def resolution_idx(self) -> int:
        return self.prev_action_idx_offset + len(self.action_list)

    @property
    def budget_idx(self) -> int:
        return self.resolution_idx + self._AUX_RESOLUTION_DIM

    @property
    def target_idx_position(self) -> int:
        return self.budget_idx + self._AUX_BUDGET_DIM

    @property
    def aux_features_dim(self) -> int:
        return (
            self._AUX_GPS_DIM
            + self._AUX_COMPASS_DIM
            + len(self.action_list)
            + self._AUX_RESOLUTION_DIM
            + self._AUX_BUDGET_DIM
            + self._AUX_TARGET_IDX_DIM
        )
    def _geodesic_distance_from(
        self,
        position: dict,
        object_id: str,
    ) -> float:
        """Geodesic (shortest navigable path) distance from an arbitrary position
        to a target object. Returns float('inf') on pathfinding failure or on
        degenerate single-corner paths.

        Cached by (grid cell, object_id). Cache is invalidated in reset().
        """
        grid = self.move_magnitude
        key = (
            round(position["x"] / grid),
            round(position["z"] / grid),
            object_id,
        )
        cached = self._geo_cache.get(key)
        if cached is not None:
            return cached

        try:
            path_event = self.controller.step(
                action="GetShortestPath",
                position=position,
                objectId=object_id,
            )
            if not path_event.metadata["lastActionSuccess"]:
                return float("inf")
            corners = path_event.metadata["actionReturn"]["corners"]
            if len(corners) < 2:
                # Degenerate path: Unity considers we're already at the target.
                # Treat as a failed pathfinding so callers can react properly.
                pass
                #print(
                #    f"[WARN] degenerate path ({len(corners)} corner(s)), "
                #    f"pos=({position['x']:.2f},{position['z']:.2f}), oid={object_id}"
                #)
                return float("inf")
            dist = sum(
                math.sqrt((a["x"] - b["x"]) ** 2 + (a["z"] - b["z"]) ** 2)
                for a, b in zip(corners[:-1], corners[1:])
            )
        except Exception:
            return float("inf")

        # Only cache finite, strictly positive distances. Zero distances signal a
        # near-degenerate planner result we don't want to memoize.
        if math.isfinite(dist) and dist > 0.0:
            self._geo_cache[key] = dist
        return dist

    def _get_geodesic_distance_to_target(self) -> float:
        """Shortest navigable path length from the current agent position
        to the target object. Returns float('inf') if pathfinding fails."""
        objects = self.current_event.metadata["objects"]
        matches = [o for o in objects if o["objectType"] == self.target_obj_type]
        if not matches:
            return float("inf")
        agent_pos = self.current_event.metadata["agent"]["position"]
        return self._geodesic_distance_from(agent_pos, matches[0]["objectId"])
    
    def _get_curriculum_start_downgrade(self, episode):
        if episode < 500:
            return 2    
        if episode < 1000:
            return 3
        if episode < 1250:
            return 4
        if episode < 1500:
            return 5
        return self.base_downgrade

    def _get_curriculum_sense_budget(self, episode):
        if episode < 500:
            return 70
        if episode < 1000:
            return 65
        if episode < 1250:
            return 60
        if episode < 7500:
            return 50
        return 45

    def _teleport_agent_at_distance(
        self,
        d_min: float = 2.0,
        d_max: float = 3.0,
        use_geodesic: bool = True,
        rng: np.random.Generator | None = None,
    ) -> bool:
        rng = rng or np.random.default_rng(
            int(self.seed) + 100_000 * int(self.env_id) + int(self._reset_count)
        )

        # 1) Get target object position + id from current metadata
        objects = self.current_event.metadata["objects"]
        matches = [o for o in objects if o["objectType"] == self.target_obj_type]
        if not matches:
            return False
        target_obj = matches[0]
        target_object_id = target_obj["objectId"]
        target_position = target_obj["position"]
        tx, tz = target_position["x"], target_position["z"]

        # 2) Get all reachable positions
        event = self.controller.step(action="GetReachablePositions")
        reachable = event.metadata["actionReturn"]
        if not reachable:
            return False

        # 3) Filter by Euclidean distance first (cheap prefilter).
        # We use a wider Euclidean band than the target geodesic band, because
        # geodesic distance is always >= Euclidean (obstacles only make paths
        # longer). A position with Euclidean < d_min cannot satisfy the geodesic
        # band, so we keep the d_min lower bound.
        candidates = [
            (p, math.sqrt((p["x"] - tx) ** 2 + (p["z"] - tz) ** 2))
            for p in reachable
        ]
        candidates = [(p, d) for p, d in candidates if d_min <= d]
        if not candidates:
            return False

        # 4) Refine with geodesic distance (true path length)
        if use_geodesic:
            refined = []
            for p, _ in candidates:
                gd = self._geodesic_distance_from(p, target_object_id)
                if d_min <= gd <= d_max:
                    refined.append((p, gd))
            if not refined:
                # No candidate satisfies the geodesic band — give up cleanly so
                # the caller can widen the band or warn loudly.
                return False
            candidates = refined
        else:
            # Pure Euclidean mode: also apply the upper bound (skipped above).
            candidates = [(p, d) for p, d in candidates if d <= d_max]
            if not candidates:
                return False

        # 5) Pick one and teleport
        chosen_pos, _ = candidates[rng.integers(len(candidates))]
        rot_y = float(rng.choice([0, 90, 180, 270]))

        self.current_event = self.controller.step(
            action="Teleport",
            position=chosen_pos,
            rotation={"x": 0, "y": rot_y, "z": 0},
            horizon=0,
            standing=True,
        )
        return self.current_event.metadata["lastActionSuccess"]

    def reset(self, scene: str, episode: int, target_obj_type: str = None,) -> torch.Tensor:
        if scene != getattr(self, "_cached_scene", None):
            self._geo_cache = {}
            self._cached_scene = scene
            
        if self.randomize_object_spawn:
            self._geo_cache = {}


        print(f"[RESET] requested={scene}, current_scene={getattr(self, 'scene', None)}")
        self.scene = scene
        self._geo_cache = {}
        self.controller.reset(scene)
        event_meta = self.controller.last_event.metadata.get("sceneName", "?")
        print(f"[RESET] after controller.reset, sceneName in metadata={event_meta}")

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

        event = self.controller.step("GetReachablePositions")
        positions = event.metadata["actionReturn"]

        if positions is None or len(positions) == 0:
            # GetReachablePositions occasionally returns None after a fresh reset
            # in AI2-THOR. Retry once after re-Initialize. If still failing, raise.
            self.current_event = self.controller.step(
                action="Initialize",
                gridSize=self.move_magnitude,
                renderImage=True,
            )
            event = self.controller.step("GetReachablePositions")
            positions = event.metadata["actionReturn"]
            if positions is None or len(positions) == 0:
                raise RuntimeError(
                    f"GetReachablePositions returned None on scene {scene!r} "
                    f"after retry. lastActionSuccess="
                    f"{event.metadata.get('lastActionSuccess')}, "
                    f"errorMessage={event.metadata.get('errorMessage')!r}"
                )

        xs = [p["x"] for p in positions]
        zs = [p["z"] for p in positions]
        self._scene_bounds = {
            "x_min": min(xs), "x_max": max(xs),
            "z_min": min(zs), "z_max": max(zs),
        }

        self._reset_count += 1

        self._step_count = 0
        self._current_action = "MoveAhead"
        self._curriculum_start_downgrade = self._get_curriculum_start_downgrade(episode)
        self._current_downgrade = 0 if self.fixed_high_res else self._curriculum_start_downgrade
        self._remaining_sensing_budget = 0 if self.fixed_high_res else self._get_curriculum_sense_budget(episode)

        if self.fixed_high_res and self._current_downgrade != 0:
            raise RuntimeError(
                f"fixed_high_res=True but reset set _current_downgrade={self._current_downgrade}"
            )
        self._last_sense_was_valid = True
        self._target_ever_visible = False
        self._path_length = 0.0
        self._done = False

        if target_obj_type:
            self.target_obj_type = target_obj_type
        else:
            self._define_target()

        if self.enforce_initial_distance:
            d_target = min(self.cfg.success_distance + 0.001 * episode, 4.0)
            d_min, d_max = d_target - 0.5, d_target + 0.5
            ok = self._teleport_agent_at_distance(
                d_min=d_min,
                d_max=d_max,
                use_geodesic=True,
            )
            if not ok:
                print(f"[WARN] teleport failed: d_min={d_min:.2f}, d_max={d_max:.2f}")
        if self.enforce_initial_distance:
            ok = self._teleport_agent_at_distance(d_min=d_min, d_max=d_max, use_geodesic=True)
            pos = self.current_event.metadata["agent"]["position"]
            geo = self._get_geodesic_distance_to_target()
            euc = self._get_min_distance_to_object(self.target_obj_type)
            print(f"[TELEPORT] ok={ok}, agent=({pos['x']:.2f},{pos['z']:.2f}), geo={geo:.2f}, euc={euc:.2f}")

        agent_position = self.current_event.metadata["agent"]["position"]
        self._agent_start = (agent_position["x"], agent_position["z"])
        self._prev_agent_xz = (agent_position["x"], agent_position["z"])

        self._closest_distance = self._get_geodesic_distance_to_target()
        if not math.isfinite(self._closest_distance):
            self._closest_distance = self._get_min_distance_to_object(self.target_obj_type)
        self._prev_distance = self._closest_distance
        self._initial_distance = self._closest_distance
        self._min_distance_seen = self._closest_distance

        self._optimal_path_length = self._compute_optimal_path_length()

        return self._compute_obs()

    def step(self, action_idx: int):
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
            self._current_downgrade = self._curriculum_start_downgrade

        # Accumulate path length on actual displacement (only successful moves).
        if action in MOVE_ACTIONS and self.current_event.metadata["lastActionSuccess"]:
            agent_pos = self.current_event.metadata["agent"]["position"]
            cur_xz = (agent_pos["x"], agent_pos["z"])
            if self._prev_agent_xz is not None:
                dx = cur_xz[0] - self._prev_agent_xz[0]
                dz = cur_xz[1] - self._prev_agent_xz[1]
                self._path_length += math.sqrt(dx * dx + dz * dz)
            self._prev_agent_xz = cur_xz

        truncated = self._fail_checker()
        obs = self._compute_obs()

        current_distance = self._get_geodesic_distance_to_target()
        if not math.isfinite(current_distance):
            # Fallback to euclidean if pathfinding failed this step
            current_distance = self._get_min_distance_to_object(self.target_obj_type)
        self._min_distance_seen = min(self._min_distance_seen, current_distance)

        task_success = self._check_success()
        # Track first time the target becomes visible (for first-visibility bonus).
        target_visible_now = self._is_target_visible()

        reward = self._compute_reward(
            truncated=truncated,
            task_success=task_success,
            current_distance=current_distance,
            target_visible_now=target_visible_now,
        )

        # Update tracker AFTER reward computation so the bonus fires exactly once.
        if target_visible_now:
            self._target_ever_visible = True

        self._prev_distance = current_distance

        terminated = action == "DONE" or (
            self.auto_success_on_goal and task_success
        )

        if self.fixed_high_res and self._current_downgrade != 0:
            raise RuntimeError(
                f"fixed_high_res=True but _current_downgrade={self._current_downgrade} "
                f"after action={action}, step={self._step_count}"
            )

        spl = self._compute_spl(task_success=task_success or self._is_episode_success_for_spl(
            terminated=terminated, truncated=truncated, task_success=task_success,
        ))

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
            "optimal_path_length": self._optimal_path_length,
            "path_length": self._path_length,
            "spl": spl,
        }
        # success conditions when min_distance is reached
        target_obj = next(
            (o for o in self.current_event.metadata["objects"]
            if o["objectType"] == self.target_obj_type),
            None
        )

        self._done = terminated or truncated
        return obs, reward, terminated, truncated, info

    def close(self):
        try:
            self.controller.stop()
        except Exception:
            pass

    def update_seed(self, seed: int):
        self.seed = seed

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

    def _is_target_visible(self) -> bool:
        """True if any instance of the target type is currently visible."""
        return any(
            o["visible"]
            for o in self.current_event.metadata["objects"]
            if o["objectType"] == self.target_obj_type
        )


    def _compute_optimal_path_length(self) -> float:
        """Shortest path length from agent start to the target object.

        Uses AI2-THOR's built-in path planner. Falls back to the initial
        euclidean distance if the planner fails (which happens on some scenes
        and configurations). The fallback is an *under-estimate* of the true
        shortest path, so it inflates SPL slightly, but it never crashes.
        """
        try:
            from ai2thor.util.metrics import get_shortest_path_to_object_type
        except Exception:
            return float(self._initial_distance)

        agent_pos = self.current_event.metadata["agent"]["position"]
        try:
            corners = get_shortest_path_to_object_type(
                controller=self.controller,
                object_type=self.target_obj_type,
                initial_position=dict(
                    x=agent_pos["x"], y=agent_pos["y"], z=agent_pos["z"]
                ),
            )
        except Exception:
            return float(self._initial_distance)

        if not corners or len(corners) < 2:
            return float(self._initial_distance)

        total = 0.0
        for a, b in zip(corners[:-1], corners[1:]):
            dx = b["x"] - a["x"]
            dz = b["z"] - a["z"]
            total += math.sqrt(dx * dx + dz * dz)
        # Clamp to a minimum to avoid division explosions.
        return max(total, 1e-3)

    def _compute_spl(self, task_success: bool) -> float:
        """SPL = success * (shortest_path / max(path, shortest_path)).

        Standard PointNav/ObjectNav metric (Anderson et al. 2018).
        """
        if not task_success:
            return 0.0
        sp = float(self._optimal_path_length)
        pl = float(self._path_length)
        if sp <= 0.0:
            return 0.0
        return sp / max(pl, sp)

    def _is_episode_success_for_spl(self, terminated: bool, truncated: bool, task_success: bool) -> bool:
        """SPL is only credited if the episode actually terminated successfully.
        Truncation (timeout) does not count, even if task_success briefly held.
        """
        if not task_success:
            return False
        if terminated:
            return True
        return False

    def _compute_reward(
        self,
        truncated: bool,
        task_success: bool | None = None,
        current_distance: float | None = None,
        target_visible_now: bool | None = None,
    ) -> float:
        if task_success is None:
            task_success = self._check_success()
        if current_distance is None:
            current_distance = self._get_min_distance_to_object(self.target_obj_type)
        if target_visible_now is None:
            target_visible_now = self._is_target_visible()

        reward = 0.0
        action = self._current_action
        cfg = self.cfg

        if cfg.use_signed_progress:
            # Bidirectional, reward = scale * (prev - curr).
            # Sums to scale * (initial_dist - final_dist) over the episode.
            if math.isfinite(self._prev_distance) and math.isfinite(current_distance):
                progress = self._prev_distance - current_distance
                reward += cfg.distance_scale * progress
        else:
            # One-sided based on best distance so far.
            progress = self._closest_distance - current_distance
            if progress > 0:
                reward += cfg.distance_scale * progress
                self._closest_distance = current_distance

        if (
            cfg.first_visibility_bonus > 0.0
            and target_visible_now
            and not self._target_ever_visible
        ):
            reward += cfg.first_visibility_bonus

        if action == "SENSE":
            reward -= cfg.sense_penalty if self._last_sense_was_valid else cfg.oversensing_penalty
        elif action == "DONE":
            if task_success:
                reward += cfg.success_reward
            else:
                dist_ratio = min(max(current_distance - self.success_distance, 0.0) / max(self.success_distance, 1e-6), 3.0)
                reward -= cfg.wrong_done_penalty * dist_ratio
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
            # o["visible"] and self._get_distance_to_position(o["position"]) <= self.success_distance
            self._get_distance_to_position(o["position"]) <= self.success_distance
            for o in targets
        )

    def get_aux_features(self, prev_action_idx: int | None = None) -> torch.Tensor:
        if self.current_event is None:
            raise RuntimeError("Environment has not been reset yet.")

        metadata = self.current_event.metadata
        agent_metadata = metadata["agent"]
        position = agent_metadata["position"]
        rotation = agent_metadata["rotation"]

        b = self._scene_bounds
        x_norm = 2 * (position["x"] - b["x_min"]) / (b["x_max"] - b["x_min"] + 1e-6) - 1
        z_norm = 2 * (position["z"] - b["z_min"]) / (b["z_max"] - b["z_min"] + 1e-6) - 1
        gps = torch.tensor([x_norm, z_norm], dtype=torch.float32, device=self.device)

        yaw_rad = math.radians(rotation["y"])
        horizon_rad = math.radians(agent_metadata.get("cameraHorizon", 0.0))
        compass = torch.tensor(
            [
                math.sin(yaw_rad), math.cos(yaw_rad),
                math.sin(horizon_rad), math.cos(horizon_rad),
            ],
            dtype=torch.float32, device=self.device,
        )

        prev_action = torch.zeros(
            len(self.action_list), dtype=torch.float32, device=self.device,
        )
        if prev_action_idx is not None:
            prev_action[prev_action_idx] = 1.0

        resolution_level = torch.tensor(
            [self._current_downgrade / max(self.base_downgrade, 1)],
            dtype=torch.float32, device=self.device,
        )

        sensing_budget = torch.tensor(
            [self._remaining_sensing_budget / max(self.max_sensing_budget, 1)],
            dtype=torch.float32, device=self.device,
        )

        target_idx = torch.tensor(
            [float(self._get_target_object_idx())],
            dtype=torch.float32, device=self.device,
        )

        return torch.cat(
            [gps, compass, prev_action, resolution_level, sensing_budget, target_idx],
            dim=0,
        )
