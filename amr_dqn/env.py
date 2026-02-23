from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path
from typing import Any

import cv2
import gym
import numpy as np

from amr_dqn.maps import MapSpec


_ACTIONS_8 = np.array(
    [
        (0, 1),  # up
        (0, -1),  # down
        (-1, 0),  # left
        (1, 0),  # right
        (1, 1),  # upper right
        (1, -1),  # lower right
        (-1, 1),  # upper left
        (-1, -1),  # lower left
    ],
    dtype=np.int32,
)


@dataclass(frozen=True)
class RewardWeights:
    lambda_target: float = 1.0
    lambda_distance: float = -0.35
    lambda_boundary: float = -1.0
    lambda_obstacle: float = -1.0


class AMRGridEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        map_spec: MapSpec,
        *,
        sensor_range: int = 6,
        max_steps: int = 500,
        reward: RewardWeights = RewardWeights(),
        cell_size: float = 1.0,
        safe_distance: float = 0.6,
        obs_map_size: int = 12,
        terminate_on_collision: bool = False,
    ) -> None:
        super().__init__()

        self.map_spec = map_spec
        self._grid = map_spec.obstacle_grid()  # (H, W), y=0 bottom
        self._height, self._width = self._grid.shape
        self.start_xy = map_spec.start_xy
        self.goal_xy = map_spec.goal_xy
        self.sensor_range = int(sensor_range)
        self.max_steps = int(max_steps)
        self.reward = reward
        self.cell_size = float(cell_size)
        if not (self.cell_size > 0):
            raise ValueError("cell_size must be > 0")
        self.safe_distance = float(safe_distance)
        self.terminate_on_collision = bool(terminate_on_collision)

        self.action_space = gym.spaces.Discrete(8)
        # Global-planning observation: full occupancy grid (downsampled) + agent/goal pose.
        self.obs_map_size = int(obs_map_size)
        if self.obs_map_size < 4:
            raise ValueError("obs_map_size must be >= 4")
        grid_ds = cv2.resize(
            self._grid.astype(np.float32, copy=False),
            dsize=(int(self.obs_map_size), int(self.obs_map_size)),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32, copy=False)
        self._obs_grid_flat = (2.0 * grid_ds.reshape(-1) - 1.0).astype(np.float32, copy=False)

        obs_dim = 5 + int(self.obs_map_size) * int(self.obs_map_size)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self._rng = np.random.default_rng()
        self._agent_xy = np.array(self.start_xy, dtype=np.int32)
        self._steps = 0
        self._dist_to_obstacle = self._compute_dist_to_obstacle()

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._agent_xy = np.array(self.start_xy, dtype=np.int32)
        self._steps = 0
        obs = self._observe()
        info = {"agent_xy": tuple(self._agent_xy.tolist())}
        return obs, info

    def step(self, action: int):
        self._steps += 1

        dx, dy = _ACTIONS_8[int(action)]
        old_xy = self._agent_xy.copy()
        new_xy = old_xy + np.array([dx, dy], dtype=np.int32)

        boundary_violation = not self._in_bounds(new_xy[0], new_xy[1])
        collision = False
        if boundary_violation:
            new_xy = old_xy  # stay in place
        else:
            collision = bool(self._grid[new_xy[1], new_xy[0]])
            if collision:
                new_xy = old_xy  # stay in place on collision

        self._agent_xy = new_xy

        reached = self._reached_goal()
        truncated = self._steps >= self.max_steps
        terminated = reached or (self.terminate_on_collision and collision)

        reward = self._reward(
            reached=reached,
            boundary_violation=boundary_violation,
            collision=collision,
        )

        obs = self._observe()
        info = {
            "agent_xy": tuple(self._agent_xy.tolist()),
            "boundary_violation": boundary_violation,
            "collision": collision,
            "reached": reached,
            "steps": self._steps,
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self._width and 0 <= y < self._height

    def _reached_goal(self) -> bool:
        return int(self._agent_xy[0]) == self.goal_xy[0] and int(self._agent_xy[1]) == self.goal_xy[1]

    def _distance_to_goal(self) -> float:
        dx = float(self._agent_xy[0] - self.goal_xy[0]) * float(self.cell_size)
        dy = float(self._agent_xy[1] - self.goal_xy[1]) * float(self.cell_size)
        return float(np.sqrt(dx * dx + dy * dy))

    def _min_obstacle_distance_cells(self) -> float:
        ax, ay = int(self._agent_xy[0]), int(self._agent_xy[1])
        return float(self._dist_to_obstacle[ay, ax])

    def _min_obstacle_distance(self) -> float:
        return float(self._min_obstacle_distance_cells()) * float(self.cell_size)

    def _compute_dist_to_obstacle(self) -> np.ndarray:
        # cv2.distanceTransform computes distance to nearest zero pixel, so we make
        # obstacles zeros and free space non-zero. Use top-left origin for OpenCV,
        # then flip back to y=0 bottom.
        grid_top = self._grid[::-1, :]
        free = (grid_top == 0).astype(np.uint8) * 255
        dist_top = cv2.distanceTransform(
            free, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE
        ).astype(np.float32)

        # Convert center-to-center distances to approximate clearance from the agent
        # (assumed at the cell center) to the obstacle cell boundary.
        dist_top = np.maximum(0.0, dist_top - 0.5).astype(np.float32, copy=False)
        return dist_top[::-1, :].astype(np.float32, copy=False)

    def _ray_distances(self) -> np.ndarray:
        ax, ay = int(self._agent_xy[0]), int(self._agent_xy[1])
        sr = self.sensor_range
        distances = np.zeros((8,), dtype=np.float32)
        for i, (dx, dy) in enumerate(_ACTIONS_8):
            d = 0
            x, y = ax, ay
            for step in range(1, sr + 1):
                x = ax + int(dx) * step
                y = ay + int(dy) * step
                if not self._in_bounds(x, y):
                    break
                if self._grid[y, x] == 1:
                    break
                d = step
            distances[i] = float(d) / float(sr)
        return distances

    def _observe(self) -> np.ndarray:
        ax, ay = int(self._agent_xy[0]), int(self._agent_xy[1])
        gx, gy = self.goal_xy
        # Normalize to [-1, 1] (stable for MLPs).
        ax_n = 2.0 * (ax / max(1, self._width - 1)) - 1.0
        ay_n = 2.0 * (ay / max(1, self._height - 1)) - 1.0
        gx_n = 2.0 * (gx / max(1, self._width - 1)) - 1.0
        gy_n = 2.0 * (gy / max(1, self._height - 1)) - 1.0

        td = self._distance_to_goal()
        diag = float(np.sqrt((self._width - 1) ** 2 + (self._height - 1) ** 2)) * float(self.cell_size)
        td01 = float(td / max(1e-6, diag))
        td_n = float(np.clip(2.0 * td01 - 1.0, -1.0, 1.0))

        obs = np.concatenate(
            [np.array([ax_n, ay_n, gx_n, gy_n, td_n], dtype=np.float32), self._obs_grid_flat],
        )
        return obs

    def _reward(self, *, reached: bool, boundary_violation: bool, collision: bool) -> float:
        r = 0.0
        if reached:
            r += float(self.reward.lambda_target)

        # Heuristic distance shaping (Eq. 11)
        r += float(self.reward.lambda_distance) * float(self._distance_to_goal())

        if boundary_violation:
            r += float(self.reward.lambda_boundary)

        # Obstacle proximity / collision penalty (Eq. 13 surrogate)
        if collision:
            r += float(self.reward.lambda_obstacle)
        else:
            od = self._min_obstacle_distance()
            if od < float(self.safe_distance):
                r += float(self.reward.lambda_obstacle)

        return float(r)


@dataclass(frozen=True)
class BicycleModelParams:
    dt: float = 0.05
    wheelbase_m: float = 0.6

    v_max_m_s: float = 2.0
    a_max_m_s2: float = 1.5

    delta_max_rad: float = math.radians(27.0)
    omega_max_rad_s: float = 1.223
    delta_dot_max_rad_s: float = math.radians(60.0)


def build_ackermann_action_table_35(*, delta_dot_max_rad_s: float, a_max_m_s2: float) -> np.ndarray:
    """Returns (35, 2) array with columns [delta_dot(rad/s), a(m/s^2)]."""
    dd = float(delta_dot_max_rad_s)
    aa = float(a_max_m_s2)
    delta_dots = np.array(
        [-dd, -(2.0 / 3.0) * dd, -(1.0 / 3.0) * dd, 0.0, (1.0 / 3.0) * dd, (2.0 / 3.0) * dd, dd],
        dtype=np.float32,
    )
    accels = np.array([-aa, -0.5 * aa, 0.0, 0.5 * aa, aa], dtype=np.float32)
    table = np.zeros((delta_dots.size * accels.size, 2), dtype=np.float32)
    k = 0
    for d_dot in delta_dots:
        for a in accels:
            table[k, 0] = float(d_dot)
            table[k, 1] = float(a)
            k += 1
    return table


def wrap_angle_rad(x: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return float((float(x) + math.pi) % (2.0 * math.pi) - math.pi)


def bicycle_integrate_one_step(
    *,
    x_m: float,
    y_m: float,
    psi_rad: float,
    v_m_s: float,
    delta_rad: float,
    delta_dot_rad_s: float,
    a_m_s2: float,
    params: BicycleModelParams,
) -> tuple[float, float, float, float, float]:
    """Rear-axle center bicycle model with one-step Euler integration."""
    dt = float(params.dt)
    v_next = float(np.clip(v_m_s + float(a_m_s2) * dt, -float(params.v_max_m_s), float(params.v_max_m_s)))

    delta_unclipped = float(delta_rad) + float(delta_dot_rad_s) * dt
    delta_lim = float(params.delta_max_rad)
    delta_next = float(np.clip(delta_unclipped, -float(delta_lim), +float(delta_lim)))

    x_next = float(x_m) + v_next * math.cos(float(psi_rad)) * dt
    y_next = float(y_m) + v_next * math.sin(float(psi_rad)) * dt
    psi_next = wrap_angle_rad(float(psi_rad) + (v_next / float(params.wheelbase_m)) * math.tan(delta_next) * dt)
    return x_next, y_next, psi_next, v_next, delta_next


def min_steps_to_cover_distance_m(
    distance_m: float,
    *,
    dt: float,
    v_max_m_s: float,
    a_max_m_s2: float,
    v0_m_s: float = 0.0,
) -> int:
    """Minimum steps needed to cover `distance_m` along a straight line.

    Uses the same discrete update as the environment speed integrator for forward motion:
        v_{k+1} = clip(v_k + a_max * dt, 0, v_max)
        x_{k+1} = x_k + v_{k+1} * dt
    """
    dist = max(0.0, float(distance_m))
    if dist <= 0.0:
        return 0

    dt_ = float(dt)
    if not (dt_ > 0.0):
        raise ValueError("dt must be > 0")
    v_max = float(v_max_m_s)
    if not (v_max > 0.0):
        raise ValueError("v_max_m_s must be > 0")
    a_max = float(a_max_m_s2)
    if not (a_max > 0.0):
        raise ValueError("a_max_m_s2 must be > 0")

    v = max(0.0, float(v0_m_s))
    covered = 0.0
    steps = 0
    while covered < dist:
        steps += 1
        v = min(v_max, v + a_max * dt_)
        covered += v * dt_
        if steps > 1_000_000:
            raise RuntimeError("min_steps_to_cover_distance_m exceeded step limit; check inputs.")
    return int(steps)


@dataclass(frozen=True)
class TwoCircleFootprint:
    radius_m: float = float(math.hypot(0.740 / 2.0, 0.924 / 4.0))
    x1_m: float = float((0.6 / 2.0) - (0.924 / 4.0))
    x2_m: float = float((0.6 / 2.0) + (0.924 / 4.0))


def compute_edt_distance_m(grid_y0_bottom: np.ndarray, *, cell_size_m: float) -> np.ndarray:
    """EDT distance (meters) from each cell center to the nearest obstacle cell center."""
    grid_top = grid_y0_bottom[::-1, :]
    free = (grid_top == 0).astype(np.uint8) * 255
    dist_top = cv2.distanceTransform(
        free, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE
    ).astype(np.float32)
    return (dist_top[::-1, :] * float(cell_size_m)).astype(np.float32, copy=False)


def bilinear_sample_2d(arr: np.ndarray, *, x: float, y: float, default: float = float("inf")) -> float:
    """Bilinear sample on a (H, W) array using x/y in index coordinates."""
    h, w = arr.shape
    if not (0.0 <= x <= (w - 1) and 0.0 <= y <= (h - 1)):
        return float(default)
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    fx = float(x - x0)
    fy = float(y - y0)

    v00 = float(arr[y0, x0])
    v10 = float(arr[y0, x1])
    v01 = float(arr[y1, x0])
    v11 = float(arr[y1, x1])
    v0 = v00 * (1.0 - fx) + v10 * fx
    v1 = v01 * (1.0 - fx) + v11 * fx
    return float(v0 * (1.0 - fy) + v1 * fy)


def bilinear_sample_2d_finite(
    arr: np.ndarray,
    *,
    x: float,
    y: float,
    fill_value: float,
    default: float | None = None,
) -> float:
    """Bilinear sample that replaces non-finite corner values with `fill_value`.

    This is important for sampling cost-to-go fields that use `inf` to mark
    non-traversable cells: plain bilinear interpolation would propagate `inf`
    into neighboring valid regions and destroy shaping gradients near obstacles.
    """

    h, w = arr.shape
    if not (0.0 <= x <= (w - 1) and 0.0 <= y <= (h - 1)):
        return float(fill_value if default is None else default)

    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    fx = float(x - x0)
    fy = float(y - y0)

    v00 = float(arr[y0, x0])
    v10 = float(arr[y0, x1])
    v01 = float(arr[y1, x0])
    v11 = float(arr[y1, x1])

    fv = float(fill_value)
    if not math.isfinite(v00):
        v00 = fv
    if not math.isfinite(v10):
        v10 = fv
    if not math.isfinite(v01):
        v01 = fv
    if not math.isfinite(v11):
        v11 = fv

    v0 = v00 * (1.0 - fx) + v10 * fx
    v1 = v01 * (1.0 - fx) + v11 * fx
    return float(v0 * (1.0 - fy) + v1 * fy)


def bilinear_sample_2d_vec(
    arr: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    default: float = float("inf"),
) -> np.ndarray:
    """Vectorized bilinear sampling on a (H, W) array using x/y in index coordinates."""
    h, w = arr.shape
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)

    if h == 0 or w == 0:
        return np.full_like(xv, float(default), dtype=np.float64)

    mask = (xv >= 0.0) & (xv <= float(w - 1)) & (yv >= 0.0) & (yv <= float(h - 1))
    x_c = np.clip(xv, 0.0, float(w - 1))
    y_c = np.clip(yv, 0.0, float(h - 1))

    x0 = np.floor(x_c).astype(np.int32, copy=False)
    y0 = np.floor(y_c).astype(np.int32, copy=False)
    x1 = np.minimum(x0 + 1, int(w - 1)).astype(np.int32, copy=False)
    y1 = np.minimum(y0 + 1, int(h - 1)).astype(np.int32, copy=False)

    fx = x_c - x0.astype(np.float64, copy=False)
    fy = y_c - y0.astype(np.float64, copy=False)

    v00 = arr[y0, x0].astype(np.float64, copy=False)
    v10 = arr[y0, x1].astype(np.float64, copy=False)
    v01 = arr[y1, x0].astype(np.float64, copy=False)
    v11 = arr[y1, x1].astype(np.float64, copy=False)

    v0 = v00 * (1.0 - fx) + v10 * fx
    v1 = v01 * (1.0 - fx) + v11 * fx
    out = v0 * (1.0 - fy) + v1 * fy
    return np.where(mask, out, float(default)).astype(np.float64, copy=False)


def bilinear_sample_2d_finite_vec(
    arr: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    fill_value: float,
    default: float | None = None,
) -> np.ndarray:
    """Vectorized bilinear sampling that replaces non-finite corner values with `fill_value`."""
    h, w = arr.shape
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)

    if h == 0 or w == 0:
        fv = float(fill_value if default is None else default)
        return np.full_like(xv, fv, dtype=np.float64)

    mask = (xv >= 0.0) & (xv <= float(w - 1)) & (yv >= 0.0) & (yv <= float(h - 1))
    x_c = np.clip(xv, 0.0, float(w - 1))
    y_c = np.clip(yv, 0.0, float(h - 1))

    x0 = np.floor(x_c).astype(np.int32, copy=False)
    y0 = np.floor(y_c).astype(np.int32, copy=False)
    x1 = np.minimum(x0 + 1, int(w - 1)).astype(np.int32, copy=False)
    y1 = np.minimum(y0 + 1, int(h - 1)).astype(np.int32, copy=False)

    fx = x_c - x0.astype(np.float64, copy=False)
    fy = y_c - y0.astype(np.float64, copy=False)

    v00 = arr[y0, x0].astype(np.float64, copy=False)
    v10 = arr[y0, x1].astype(np.float64, copy=False)
    v01 = arr[y1, x0].astype(np.float64, copy=False)
    v11 = arr[y1, x1].astype(np.float64, copy=False)

    fv = float(fill_value)
    v00 = np.where(np.isfinite(v00), v00, fv)
    v10 = np.where(np.isfinite(v10), v10, fv)
    v01 = np.where(np.isfinite(v01), v01, fv)
    v11 = np.where(np.isfinite(v11), v11, fv)

    v0 = v00 * (1.0 - fx) + v10 * fx
    v1 = v01 * (1.0 - fx) + v11 * fx
    out = v0 * (1.0 - fy) + v1 * fy

    outside = float(fill_value if default is None else default)
    return np.where(mask, out, outside).astype(np.float64, copy=False)


def dijkstra_cost_to_goal_m(
    traversable_y0_bottom: np.ndarray,
    *,
    goal_xy: tuple[int, int],
    cell_size_m: float,
) -> np.ndarray:
    """Compute an 8-connected shortest-path cost-to-go field (meters) via Dijkstra."""
    if traversable_y0_bottom.ndim != 2:
        raise ValueError("traversable_y0_bottom must be a 2D array")
    h, w = traversable_y0_bottom.shape
    if h == 0 or w == 0:
        raise ValueError("traversable_y0_bottom must be non-empty")
    cell = float(cell_size_m)
    if not (cell > 0.0):
        raise ValueError("cell_size_m must be > 0")

    gx, gy = int(goal_xy[0]), int(goal_xy[1])
    if not (0 <= gx < w and 0 <= gy < h):
        raise ValueError("goal_xy is out of bounds")

    traversable = traversable_y0_bottom.astype(bool, copy=False)

    cost = np.full((h, w), float("inf"), dtype=np.float64)
    if not bool(traversable[gy, gx]):
        return cost

    pq: list[tuple[float, int, int]] = []
    cost[gy, gx] = 0.0
    heapq.heappush(pq, (0.0, gx, gy))

    moves: tuple[tuple[int, int, float], ...] = (
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, 1, 1.0),
        (0, -1, 1.0),
        (1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (-1, -1, math.sqrt(2.0)),
    )

    while pq:
        d, x, y = heapq.heappop(pq)
        if d != float(cost[y, x]):
            continue
        for dx, dy, step in moves:
            nx = x + dx
            ny = y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if not bool(traversable[ny, nx]):
                continue
            nd = float(d) + float(step) * cell
            if nd < float(cost[ny, nx]):
                cost[ny, nx] = float(nd)
                heapq.heappush(pq, (float(nd), nx, ny))

    return cost.astype(np.float32, copy=False)


class AMRBicycleEnv(gym.Env):
    """Ackermann/bicycle dynamics on a grid occupancy map using EDT for collision + clearance (OD)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        map_spec: MapSpec,
        *,
        max_steps: int = 500,
        cell_size_m: float = 0.1,
        model: BicycleModelParams = BicycleModelParams(),
        footprint: TwoCircleFootprint = TwoCircleFootprint(),
        sensor_range_m: float = 6.0,
        n_sectors: int = 36,
        obs_map_size: int = 12,
        od_cap_m: float = 2.0,
        safe_distance_m: float = 0.20,
        safe_speed_distance_m: float = 0.20,
        # A slightly looser positional tolerance improves robustness with discrete controls
        # and short-horizon safety shields (the paper's 0.30m can be hard to hit exactly).
        goal_tolerance_m: float = 1,
        goal_angle_tolerance_deg: float = 180.0,
        reward_k_p: float = 12.0,
        reward_k_t: float = 0.1,
        reward_k_delta: float = 1.5,
        reward_k_a: float = 0.2,
        reward_k_kappa: float = 0.2,
        reward_k_o: float = 1.5,
        reward_k_v: float = 2.0,
        reward_k_c: float = 0.0,
        reward_obs_max: float = 10.0,
        stuck_steps: int = 20,
        stuck_min_disp_m: float = 0.02,
        stuck_min_speed_m_s: float = 0.05,
        stuck_penalty: float = 300.0,
    ) -> None:
        super().__init__()

        self.map_spec = map_spec
        self._grid = map_spec.obstacle_grid().astype(np.uint8, copy=False)  # (H, W), y=0 bottom
        self._height, self._width = self._grid.shape
        self._canonical_start_xy = (int(map_spec.start_xy[0]), int(map_spec.start_xy[1]))
        self._canonical_goal_xy = (int(map_spec.goal_xy[0]), int(map_spec.goal_xy[1]))
        self.start_xy = (int(self._canonical_start_xy[0]), int(self._canonical_start_xy[1]))
        self.goal_xy = (int(self._canonical_goal_xy[0]), int(self._canonical_goal_xy[1]))

        self.max_steps = int(max_steps)
        self.cell_size_m = float(cell_size_m)
        if not (self.cell_size_m > 0):
            raise ValueError("cell_size_m must be > 0")

        self.model = model
        self.footprint = footprint
        self.sensor_range_m = float(sensor_range_m)
        if not (self.sensor_range_m > 0):
            raise ValueError("sensor_range_m must be > 0")
        self.n_sectors = int(n_sectors)
        if self.n_sectors < 1:
            raise ValueError("n_sectors must be >= 1")
        self.obs_map_size = int(obs_map_size)
        if self.obs_map_size < 4:
            raise ValueError("obs_map_size must be >= 4")
        self.od_cap_m = float(od_cap_m)
        if not (self.od_cap_m > 0):
            raise ValueError("od_cap_m must be > 0")
        self.safe_distance_m = float(safe_distance_m)
        if not (self.safe_distance_m > 0):
            raise ValueError("safe_distance_m must be > 0")
        self.safe_speed_distance_m = float(safe_speed_distance_m)
        if not (self.safe_speed_distance_m > 0):
            raise ValueError("safe_speed_distance_m must be > 0")
        if float(self.safe_speed_distance_m) < float(self.safe_distance_m):
            raise ValueError("safe_speed_distance_m must be >= safe_distance_m")
        self.goal_tolerance_m = float(goal_tolerance_m)
        if not (self.goal_tolerance_m > 0):
            raise ValueError("goal_tolerance_m must be > 0")
        self.goal_angle_tolerance_rad = float(math.radians(float(goal_angle_tolerance_deg)))
        if not (0.0 < self.goal_angle_tolerance_rad <= math.pi):
            raise ValueError("goal_angle_tolerance_deg must be in (0, 180]")

        # Precompute EDT + cost-to-go once (forest maps are static).
        self._eps_cell_m = float(math.sqrt(2.0) * 0.5 * self.cell_size_m)
        self._half_cell_m = float(0.5 * self.cell_size_m)
        self._dist_m = compute_edt_distance_m(self._grid, cell_size_m=self.cell_size_m)
        self._diag_m = float(
            math.hypot(float(self._width - 1) * self.cell_size_m, float(self._height - 1) * self.cell_size_m)
        )

        # Treat the world boundary as an obstacle for both collision checking and sensing.
        max_x = float(self._width - 1) * self.cell_size_m
        max_y = float(self._height - 1) * self.cell_size_m
        xs = (np.arange(self._width, dtype=np.float32) * float(self.cell_size_m)).reshape(1, -1)
        ys = (np.arange(self._height, dtype=np.float32) * float(self.cell_size_m)).reshape(-1, 1)
        boundary_dist = np.minimum(
            np.minimum(xs, float(max_x) - xs),
            np.minimum(ys, float(max_y) - ys),
        ).astype(np.float32, copy=False)
        self._dist_m = np.minimum(self._dist_m, boundary_dist).astype(np.float32, copy=False)

        # Traversability used for cost-to-go shaping and curriculum sampling.
        #
        # Use *collision-free* clearance (r + eps_cell). The reward's OD-based safe-distance terms
        # handle additional margin; making the cost-to-go field too conservative can disconnect the
        # free space and remove useful progress gradients.
        self._clearance_thr_m = float(self.footprint.radius_m) + float(self._eps_cell_m)
        self._traversable_base = (self._dist_m > float(self._clearance_thr_m)).astype(bool, copy=False)
        # Ensure the canonical start/goal cells are always treated as traversable.
        self._traversable_base[self._canonical_start_xy[1], self._canonical_start_xy[0]] = True
        self._traversable_base[self._canonical_goal_xy[1], self._canonical_goal_xy[0]] = True

        # Candidate free cells for random start/goal sampling.
        free_y, free_x = np.where(self._traversable_base)
        self._rand_free_xy = np.stack([free_x, free_y], axis=1).astype(np.int32, copy=False)

        # Goal-dependent fields (cost-to-go + curriculum candidates).
        self._set_goal_xy(self.goal_xy)
        # Start-dependent normalization + downsampled cost map.
        self._update_start_dependent_fields(start_xy=self.start_xy)

        # Sanity-check horizon: use cost-to-go from start (accounts for detours).
        min_steps = min_steps_to_cover_distance_m(
            max(0.0, float(self._cost_norm_m) - float(self.goal_tolerance_m)),
            dt=float(self.model.dt),
            v_max_m_s=float(self.model.v_max_m_s),
            a_max_m_s2=float(self.model.a_max_m_s2),
            v0_m_s=0.0,
        )
        if int(self.max_steps) < int(min_steps):
            raise ValueError(
                f"max_steps={self.max_steps} is too small for forest env {self.map_spec.name!r} "
                f"(cost-to-go≈{self._cost_norm_m:.2f}m with v_max={self.model.v_max_m_s:.2f}m/s, "
                f"a_max={self.model.a_max_m_s2:.2f}m/s², dt={self.model.dt:.3f}s). "
                f"Need at least {min_steps} steps (increase --max-steps)."
            )

        self.reward_k_p = float(reward_k_p)
        self.reward_k_t = float(reward_k_t)
        self.reward_k_delta = float(reward_k_delta)
        self.reward_k_a = float(reward_k_a)
        self.reward_k_kappa = float(reward_k_kappa)
        self.reward_k_o = float(reward_k_o)
        self.reward_k_v = float(reward_k_v)
        self.reward_k_c = float(reward_k_c)
        self.reward_obs_max = float(reward_obs_max)
        self.reward_eps = 1e-3
        self.stuck_steps = int(stuck_steps)
        if self.stuck_steps < 1:
            raise ValueError("stuck_steps must be >= 1")
        self.stuck_min_disp_m = float(stuck_min_disp_m)
        if not (self.stuck_min_disp_m >= 0.0):
            raise ValueError("stuck_min_disp_m must be >= 0")
        self.stuck_min_speed_m_s = float(stuck_min_speed_m_s)
        if not (self.stuck_min_speed_m_s >= 0.0):
            raise ValueError("stuck_min_speed_m_s must be >= 0")
        self.stuck_penalty = float(stuck_penalty)
        if not (self.stuck_penalty >= 0.0):
            raise ValueError("stuck_penalty must be >= 0")

        self.action_table = build_ackermann_action_table_35(
            delta_dot_max_rad_s=float(model.delta_dot_max_rad_s),
            a_max_m_s2=float(model.a_max_m_s2),
        )
        self.action_space = gym.spaces.Discrete(int(self.action_table.shape[0]))

        # Global-planning observation: agent/goal pose + downsampled (static) maps.
        #
        # The obstacle grid and cost-to-go field are known in global planning and provide
        # much richer context than sensor-only lidar features.
        occ_ds = cv2.resize(
            self._grid.astype(np.float32, copy=False),
            dsize=(int(self.obs_map_size), int(self.obs_map_size)),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32, copy=False)
        self._obs_occ_flat = (2.0 * occ_ds.reshape(-1) - 1.0).astype(np.float32, copy=False)

        cost = np.minimum(self._cost_to_goal_m, float(self._cost_fill_m)).astype(np.float32, copy=False)
        cost01 = np.clip(cost / max(1e-6, float(self._cost_norm_m)), 0.0, 1.0).astype(np.float32, copy=False)
        cost_ds = cv2.resize(
            cost01,
            dsize=(int(self.obs_map_size), int(self.obs_map_size)),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32, copy=False)
        self._obs_cost_flat = (2.0 * cost_ds.reshape(-1) - 1.0).astype(np.float32, copy=False)

        obs_dim = 11 + 2 * int(self.obs_map_size) * int(self.obs_map_size)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self._rng = np.random.default_rng()
        self._steps = 0

        self._x_m = float(self.start_xy[0]) * self.cell_size_m
        self._y_m = float(self.start_xy[1]) * self.cell_size_m
        self._psi_rad = 0.0
        self._v_m_s = 0.0
        self._delta_rad = 0.0
        self._prev_delta_dot = 0.0
        self._prev_a = 0.0
        self._last_od_m = 0.0
        self._last_collision = False
        self._stuck_pos_history: list[tuple[float, float]] = []
        self._ha_path_cache: dict[tuple[int, int, int, int], list[tuple[float, float]]] = {}
        self._ha_progress_idx: int = 0
        self._ha_start_xy: tuple[int, int] = self.start_xy

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    def _in_bounds_xy(self, xy: tuple[int, int]) -> bool:
        x, y = int(xy[0]), int(xy[1])
        return 0 <= x < int(self._width) and 0 <= y < int(self._height)

    def _set_goal_xy(self, goal_xy: tuple[int, int]) -> None:
        gx, gy = int(goal_xy[0]), int(goal_xy[1])
        if not self._in_bounds_xy((gx, gy)):
            raise ValueError("goal_xy is out of bounds")

        self.goal_xy = (int(gx), int(gy))

        traversable = self._traversable_base
        if not bool(traversable[int(gy), int(gx)]):
            trav = traversable.astype(bool, copy=True)
            trav[int(gy), int(gx)] = True
        else:
            trav = traversable

        self._cost_to_goal_m = dijkstra_cost_to_goal_m(
            trav,
            goal_xy=self.goal_xy,
            cell_size_m=self.cell_size_m,
        )
        finite_cost = self._cost_to_goal_m[np.isfinite(self._cost_to_goal_m)]
        if finite_cost.size == 0:
            raise ValueError(
                f"Forest map {self.map_spec.name!r} has no reachable states for goal={self.goal_xy}; "
                "pick a different goal or regenerate the map."
            )
        self._cost_fill_m = float(np.max(finite_cost)) + float(self.cell_size_m)

        # Curriculum: candidate start cells (reachable under clearance + have finite cost-to-go).
        self._curriculum_min_cost_m = float(max(self.goal_tolerance_m + self.cell_size_m, 1.0))
        cand_mask = np.isfinite(self._cost_to_goal_m) & (self._dist_m > float(self._clearance_thr_m))
        # Exclude the goal cell (too trivial) and any cells inside the goal tolerance.
        cand_mask[int(gy), int(gx)] = False
        cand_mask &= self._cost_to_goal_m >= float(self._curriculum_min_cost_m)

        cand_y, cand_x = np.where(cand_mask)
        self._curriculum_start_xy = np.stack([cand_x, cand_y], axis=1).astype(np.int32, copy=False)
        self._curriculum_start_costs_m = self._cost_to_goal_m[cand_y, cand_x].astype(np.float32, copy=False)

    def _update_start_dependent_fields(self, *, start_xy: tuple[int, int]) -> None:
        sx, sy = int(start_xy[0]), int(start_xy[1])
        if not self._in_bounds_xy((sx, sy)):
            raise ValueError("start_xy is out of bounds")

        # Normalize cost-to-go by the *anchor* start pose (two-circle footprint),
        # not just the rear-axle cell, so shaping remains meaningful near obstacles.
        start_cost_cell = float(self._cost_to_goal_m[int(sy), int(sx)])
        start_x_m = float(sx) * self.cell_size_m
        start_y_m = float(sy) * self.cell_size_m
        dx0 = float(self.goal_xy[0] - int(sx)) * self.cell_size_m
        dy0 = float(self.goal_xy[1] - int(sy)) * self.cell_size_m
        psi0 = wrap_angle_rad(math.atan2(dy0, dx0))
        start_cost_pose = self._cost_to_goal_pose_m(start_x_m, start_y_m, psi0)
        self._cost_norm_m = float(max(self._diag_m, start_cost_cell, start_cost_pose))
        if not math.isfinite(self._cost_norm_m):
            raise ValueError(
                f"Forest map {self.map_spec.name!r} is unreachable under the clearance constraint; "
                "regenerate the map or reduce obstacle density."
            )

        # Curriculum cost anchor (used only for curriculum sampling by cost bands).
        start_cost = float(start_cost_pose if math.isfinite(start_cost_pose) else start_cost_cell)
        self._curriculum_start_cost_m = float(start_cost)

        # Downsampled cost-to-go field (normalized) for global-map observations.
        cost = np.minimum(self._cost_to_goal_m, float(self._cost_fill_m)).astype(np.float32, copy=False)
        cost01 = np.clip(cost / max(1e-6, float(self._cost_norm_m)), 0.0, 1.0).astype(np.float32, copy=False)
        cost_ds = cv2.resize(
            cost01,
            dsize=(int(self.obs_map_size), int(self.obs_map_size)),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32, copy=False)
        self._obs_cost_flat = (2.0 * cost_ds.reshape(-1) - 1.0).astype(np.float32, copy=False)

    def _sample_random_start_goal(
        self,
        *,
        min_cost_m: float,
        max_cost_m: float | None,
        fixed_prob: float,
        tries: int,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        if self._rand_free_xy.size == 0:
            return self._canonical_start_xy, self._canonical_goal_xy

        p_fixed = float(np.clip(float(fixed_prob), 0.0, 1.0))
        if float(self._rng.random()) < p_fixed:
            self._set_goal_xy(self._canonical_goal_xy)
            self._update_start_dependent_fields(start_xy=self._canonical_start_xy)
            return self._canonical_start_xy, self._canonical_goal_xy

        min_cost = max(0.0, float(min_cost_m))
        max_cost = None if max_cost_m is None else max(0.0, float(max_cost_m))
        n_tries = max(1, int(tries))

        for _ in range(n_tries):
            gi = int(self._rng.integers(0, int(self._rand_free_xy.shape[0])))
            gx, gy = (int(self._rand_free_xy[gi, 0]), int(self._rand_free_xy[gi, 1]))

            try:
                self._set_goal_xy((gx, gy))
            except Exception:
                continue

            costs = self._cost_to_goal_m
            cand_mask = np.isfinite(costs) & (self._dist_m > float(self._clearance_thr_m))
            cand_mask[int(gy), int(gx)] = False
            if min_cost > 0.0:
                cand_mask &= costs >= float(min_cost)
            if max_cost is not None and max_cost > 0.0:
                cand_mask &= costs <= float(max_cost)

            sy, sx = np.where(cand_mask)
            if sx.size == 0:
                continue

            si = int(self._rng.integers(0, int(sx.size)))
            start_xy = (int(sx[si]), int(sy[si]))

            # Verify the initial pose (heading to goal) is collision-free under the two-circle footprint.
            dx0 = float(gx - int(start_xy[0])) * float(self.cell_size_m)
            dy0 = float(gy - int(start_xy[1])) * float(self.cell_size_m)
            psi0 = wrap_angle_rad(math.atan2(dy0, dx0))
            _od0, coll0 = self._od_and_collision_at_pose_m(
                float(start_xy[0]) * float(self.cell_size_m),
                float(start_xy[1]) * float(self.cell_size_m),
                float(psi0),
            )
            if bool(coll0):
                continue

            # Update normalization anchor to the sampled start.
            try:
                self._update_start_dependent_fields(start_xy=start_xy)
            except Exception:
                continue

            # Reject pairs that cannot be executed within the episode horizon.
            try:
                min_steps = min_steps_to_cover_distance_m(
                    max(0.0, float(self._cost_norm_m) - float(self.goal_tolerance_m)),
                    dt=float(self.model.dt),
                    v_max_m_s=float(self.model.v_max_m_s),
                    a_max_m_s2=float(self.model.a_max_m_s2),
                    v0_m_s=0.0,
                )
            except Exception:
                continue
            if int(self.max_steps) < int(min_steps):
                continue

            return start_xy, (gx, gy)

        # Fallback: canonical pair.
        self._set_goal_xy(self._canonical_goal_xy)
        self._update_start_dependent_fields(start_xy=self._canonical_start_xy)
        return self._canonical_start_xy, self._canonical_goal_xy

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._steps = 0

        # Default: canonical fixed start/goal (backwards compatible).
        start_xy = (int(self._canonical_start_xy[0]), int(self._canonical_start_xy[1]))
        goal_xy = (int(self._canonical_goal_xy[0]), int(self._canonical_goal_xy[1]))

        start_override: tuple[int, int] | None = None
        goal_override: tuple[int, int] | None = None
        random_start_goal = False
        rand_min_cost_m = 0.0
        rand_max_cost_m: float | None = None
        rand_fixed_prob = 0.0
        rand_tries = 200

        if options:
            if options.get("start_xy") is not None:
                sx, sy = options["start_xy"]
                start_override = (int(sx), int(sy))
            if options.get("goal_xy") is not None:
                gx, gy = options["goal_xy"]
                goal_override = (int(gx), int(gy))
            random_start_goal = bool(options.get("random_start_goal", False))
            rand_min_cost_m = float(options.get("rand_min_cost_m", 0.0))
            max_raw = options.get("rand_max_cost_m", None)
            rand_max_cost_m = None if max_raw is None else float(max_raw)
            rand_fixed_prob = float(options.get("rand_fixed_prob", 0.0))
            rand_tries = int(options.get("rand_tries", 200))

        if goal_override is not None:
            goal_xy = (int(goal_override[0]), int(goal_override[1]))

        if random_start_goal and start_override is None and goal_override is None:
            start_xy, goal_xy = self._sample_random_start_goal(
                min_cost_m=float(rand_min_cost_m),
                max_cost_m=rand_max_cost_m,
                fixed_prob=float(rand_fixed_prob),
                tries=int(rand_tries),
            )
        else:
            # Ensure env goal-dependent fields match the requested goal.
            if (int(self.goal_xy[0]), int(self.goal_xy[1])) != (int(goal_xy[0]), int(goal_xy[1])):
                self._set_goal_xy(goal_xy)

            if start_override is not None:
                start_xy = (int(start_override[0]), int(start_override[1]))
                # For explicit (start,goal) overrides, normalize the cost-to-go field by the episode start.
                self._update_start_dependent_fields(start_xy=start_xy)
            else:
                # Keep normalization anchored at the canonical start (same behavior as fixed-start training),
                # even when curriculum samples a different episode start.
                self._update_start_dependent_fields(start_xy=self._canonical_start_xy)

        self.start_xy = (int(start_xy[0]), int(start_xy[1]))
        self.goal_xy = (int(goal_xy[0]), int(goal_xy[1]))

        ha_start_xy = (int(self.start_xy[0]), int(self.start_xy[1]))
        ha_progress_idx = 0
        psi_override: float | None = None
        if (
            options
            and (not bool(random_start_goal))
            and start_override is None
            and goal_override is None
            and options.get("curriculum_progress") is not None
            and len(self._curriculum_start_xy) > 0
        ):
            # Forest curriculum: early episodes start closer to the goal; later episodes gradually
            # shift probability mass back to the canonical start. This prevents a train/test mismatch
            # where the agent never practices the true start state but inference always begins there.
            p = float(options["curriculum_progress"])
            p = float(np.clip(p, 0.0, 1.0))

            # With probability p, use the fixed start (p=1 => always start from SP).
            if float(self._rng.random()) >= float(p):
                band_m = float(options.get("curriculum_band_m", 2.0))
                band_m = max(float(self.cell_size_m), float(band_m))

                hi = float(self._curriculum_min_cost_m) + p * float(
                    max(0.0, float(self._curriculum_start_cost_m) - float(self._curriculum_min_cost_m))
                )
                lo = max(float(self._curriculum_min_cost_m), float(hi) - float(band_m))

                # Prefer sampling starts along the (precomputed) Hybrid A* reference path when available.
                # This keeps curriculum starts on a known feasible corridor and avoids repeatedly
                # re-planning from many random start states (which can time out on large forests).
                chosen_ref_idx: int | None = None
                ref_path = self._hybrid_astar_path(start_xy=self.start_xy)
                if len(ref_path) >= 2:
                    # Sample by reference-path progress (more robust than matching exact cost bands
                    # after rounding continuous Hybrid A* coordinates back to grid cells).
                    max_idx = max(0, int(len(ref_path) - 2))  # exclude last point (goal vicinity)
                    band_steps = max(1, int(round(float(band_m) / float(self.cell_size_m))))
                    target_idx = int(round((1.0 - float(p)) * float(max_idx)))
                    lo_i = max(0, int(target_idx) - int(band_steps))
                    hi_i = min(int(max_idx), int(target_idx) + int(band_steps))

                    ref_idxs: list[int] = []
                    for i in range(int(lo_i), int(hi_i) + 1):
                        px, py = ref_path[int(i)]
                        ix = int(round(float(px)))
                        iy = int(round(float(py)))
                        if not (0 <= ix < self._width and 0 <= iy < self._height):
                            continue
                        c = float(self._cost_to_goal_m[iy, ix])
                        if not math.isfinite(c) or float(c) < float(self._curriculum_min_cost_m):
                            continue
                        ref_idxs.append(int(i))

                    if ref_idxs:
                        chosen_ref_idx = int(self._rng.choice(ref_idxs))
                        px, py = ref_path[int(chosen_ref_idx)]
                        start_xy = (int(round(float(px))), int(round(float(py))))
                        ha_start_xy = (int(self.start_xy[0]), int(self.start_xy[1]))
                        ha_progress_idx = int(chosen_ref_idx)
                        j = min(int(chosen_ref_idx) + 1, len(ref_path) - 1)
                        px2, py2 = ref_path[int(j)]
                        dx = (float(px2) - float(px)) * float(self.cell_size_m)
                        dy = (float(py2) - float(py)) * float(self.cell_size_m)
                        if abs(float(dx)) + abs(float(dy)) > 1e-9:
                            psi_override = wrap_angle_rad(math.atan2(float(dy), float(dx)))

                if chosen_ref_idx is None:
                    costs = self._curriculum_start_costs_m
                    idxs = np.nonzero((costs >= float(lo)) & (costs <= float(hi)))[0]
                    if idxs.size == 0:
                        idxs = np.nonzero(costs <= float(hi))[0]
                    if idxs.size > 0:
                        j = int(self._rng.choice(idxs))
                        start_xy = (int(self._curriculum_start_xy[j, 0]), int(self._curriculum_start_xy[j, 1]))
                        ha_start_xy = (int(start_xy[0]), int(start_xy[1]))
                        ha_progress_idx = 0

        # Finalize episode start (may differ from the canonical start when curriculum/randomization is used).
        self.start_xy = (int(start_xy[0]), int(start_xy[1]))

        self._x_m = float(start_xy[0]) * self.cell_size_m
        self._y_m = float(start_xy[1]) * self.cell_size_m
        dx = float(self.goal_xy[0] - start_xy[0]) * self.cell_size_m
        dy = float(self.goal_xy[1] - start_xy[1]) * self.cell_size_m
        psi = wrap_angle_rad(math.atan2(dy, dx))
        if psi_override is not None:
            psi = float(psi_override)
        self._psi_rad = float(psi)
        self._v_m_s = 0.0
        self._delta_rad = 0.0
        self._prev_delta_dot = 0.0
        self._prev_a = 0.0
        self._last_od_m, self._last_collision = self._od_and_collision_m()
        self._stuck_pos_history = [(float(self._x_m), float(self._y_m))]
        self._ha_start_xy = (int(ha_start_xy[0]), int(ha_start_xy[1]))
        self._ha_progress_idx = int(ha_progress_idx)

        obs = self._observe()
        info = {"agent_xy": self._agent_xy_for_plot(), "pose_m": (self._x_m, self._y_m, self._psi_rad)}
        return obs, info

    def _step_with_controls(self, *, delta_dot: float, a: float):
        self._steps += 1

        delta_dot = float(delta_dot)
        a = float(a)
        prev_delta_dot = float(self._prev_delta_dot)
        prev_a = float(self._prev_a)

        x_before = float(self._x_m)
        y_before = float(self._y_m)
        # Progress shaping uses a clearance-aware cost-to-go field (helps detours around obstacles).
        cost_before = self._cost_to_goal_pose_m(x_before, y_before, float(self._psi_rad))
        d_goal_before = self._distance_to_goal_m()
        delta_before = float(self._delta_rad)

        x_next, y_next, psi_next, v_next, delta_next = bicycle_integrate_one_step(
            x_m=self._x_m,
            y_m=self._y_m,
            psi_rad=self._psi_rad,
            v_m_s=self._v_m_s,
            delta_rad=self._delta_rad,
            delta_dot_rad_s=delta_dot,
            a_m_s2=a,
            params=self.model,
        )

        self._x_m, self._y_m, self._psi_rad, self._v_m_s, self._delta_rad = (
            x_next,
            y_next,
            psi_next,
            v_next,
            delta_next,
        )
        self._last_od_m, self._last_collision = self._od_and_collision_m()

        cost_after = self._cost_to_goal_pose_m(self._x_m, self._y_m, float(self._psi_rad))
        d_goal_after = self._distance_to_goal_m()
        alpha = self._goal_relative_angle_rad()
        reached = (d_goal_after <= self.goal_tolerance_m) and (abs(alpha) <= self.goal_angle_tolerance_rad)

        collision = bool(self._last_collision)
        od_m = float(self._last_od_m)
        truncated = self._steps >= self.max_steps
        terminated = bool(collision or reached)

        # Stuck detection (helps prevent in-place steering jitter / stopping forever).
        #
        # Use *windowed* displacement, not per-step displacement: with dt=0.05s the vehicle can
        # legitimately move <2cm per step at low speeds, so per-step thresholds cause false stuck.
        stuck = False
        if not (terminated or truncated):
            self._stuck_pos_history.append((float(self._x_m), float(self._y_m)))
            max_hist = int(self.stuck_steps) + 1
            if len(self._stuck_pos_history) > max_hist:
                self._stuck_pos_history.pop(0)

            if len(self._stuck_pos_history) >= max_hist and abs(float(self._v_m_s)) < float(self.stuck_min_speed_m_s):
                x0, y0 = self._stuck_pos_history[0]
                x1, y1 = self._stuck_pos_history[-1]
                disp = float(math.hypot(float(x1) - float(x0), float(y1) - float(y0)))
                if disp < float(self.stuck_min_disp_m):
                    stuck = True
                    terminated = True

        reward = 0.0
        # Progress (short)
        if math.isfinite(cost_before) and math.isfinite(cost_after):
            reward += self.reward_k_p * float(cost_before - cost_after)
        else:
            reward += self.reward_k_p * float(d_goal_before - d_goal_after)
        # Time (fast): per-step penalty.
        reward -= self.reward_k_t
        # Smoothness
        reward -= self.reward_k_delta * float(delta_next - delta_before) ** 2
        # Acceleration smoothness should not prevent "getting going" from rest; scale by speed.
        v_scale = (float(v_next) / float(self.model.v_max_m_s)) ** 2
        reward -= self.reward_k_a * float(a - prev_a) ** 2 * float(v_scale)
        # Curvature / large steering penalty
        reward -= self.reward_k_kappa * float(math.tan(delta_next) ** 2)
        # Clearance-based safety shaping. Skip when already in collision to avoid compounding huge penalties.
        if not collision:
            od_pos = max(0.0, float(od_m))

            # Near-obstacle penalty (using OD).
            if od_pos < self.safe_distance_m:
                obs_term = (1.0 / (od_pos + self.reward_eps)) - (1.0 / (self.safe_distance_m + self.reward_eps))
                obs_pen = float(self.reward_k_o) * float(obs_term)
                reward -= min(float(self.reward_obs_max), float(obs_pen))

            # Forest near-obstacle speed coupling + optional soft speed cap.
            if od_pos < self.safe_speed_distance_m:
                # Speed coupling term (penalize speed when clearance is small).
                reward -= self.reward_k_v * ((self.safe_speed_distance_m - od_pos) / self.safe_speed_distance_m) * (
                    float(v_next) / float(self.model.v_max_m_s)
                ) ** 2

                # Soft speed cap (optional, but stabilizes forest driving in thin corridors).
                v_cap = float(self.model.v_max_m_s) * float(
                    np.clip(float(od_pos) / float(self.safe_speed_distance_m), 0.0, 1.0)
                )
                dv = max(0.0, float(v_next) - float(v_cap))
                reward -= self.reward_k_c * float(dv) ** 2

        # Terminal
        if collision:
            reward -= 200.0
        elif reached:
            reward += 400.0
        if stuck:
            reward -= float(self.stuck_penalty)

        # Markov: the *next* state carries previous action = action taken now.
        self._prev_delta_dot = float(delta_dot)
        self._prev_a = float(a)

        obs = self._observe()
        info = {
            "agent_xy": self._agent_xy_for_plot(),
            "pose_m": (self._x_m, self._y_m, self._psi_rad),
            "collision": bool(collision),
            "reached": bool(reached),
            "stuck": bool(stuck),
            "od_m": float(od_m),
            "d_goal_m": float(d_goal_after),
            "alpha_rad": float(alpha),
            "v_m_s": float(self._v_m_s),
            "delta_rad": float(self._delta_rad),
            "steps": int(self._steps),
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    def step(self, action: int):
        a_id = int(action)
        delta_dot = float(self.action_table[a_id, 0])
        a = float(self.action_table[a_id, 1])
        return self._step_with_controls(delta_dot=delta_dot, a=a)

    def step_continuous(self, *, delta_dot_rad_s: float, a_m_s2: float):
        """Continuous-control variant of `step()` (uses the same dynamics/collision/termination).

        This is intended for evaluating continuous controllers (e.g., MPC) on the forest env without
        forcing them through the discrete `action_table` interface used by DQN.
        """
        dd_max = float(self.model.delta_dot_max_rad_s)
        a_max = float(self.model.a_max_m_s2)
        delta_dot = float(np.clip(float(delta_dot_rad_s), -dd_max, +dd_max))
        a = float(np.clip(float(a_m_s2), -a_max, +a_max))
        return self._step_with_controls(delta_dot=delta_dot, a=a)

    def _agent_xy_for_plot(self) -> tuple[float, float]:
        return (float(self._x_m) / self.cell_size_m, float(self._y_m) / self.cell_size_m)

    def _circle_centers_m(self) -> tuple[tuple[float, float], tuple[float, float]]:
        c = math.cos(float(self._psi_rad))
        s = math.sin(float(self._psi_rad))
        c1 = (float(self._x_m) + c * float(self.footprint.x1_m), float(self._y_m) + s * float(self.footprint.x1_m))
        c2 = (float(self._x_m) + c * float(self.footprint.x2_m), float(self._y_m) + s * float(self.footprint.x2_m))
        return c1, c2

    def _dist_at_m(self, x_m: float, y_m: float) -> float:
        xi = float(x_m) / self.cell_size_m
        yi = float(y_m) / self.cell_size_m
        return bilinear_sample_2d(self._dist_m, x=xi, y=yi, default=0.0)

    @staticmethod
    def _wrap_angle_rad_np(x: np.ndarray) -> np.ndarray:
        return (np.remainder(np.asarray(x, dtype=np.float64) + math.pi, 2.0 * math.pi) - math.pi).astype(
            np.float64, copy=False
        )

    def _dist_at_m_vec(self, x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
        xi = np.asarray(x_m, dtype=np.float64) / float(self.cell_size_m)
        yi = np.asarray(y_m, dtype=np.float64) / float(self.cell_size_m)
        return bilinear_sample_2d_vec(self._dist_m, x=xi, y=yi, default=0.0)

    def _od_and_collision_m(self) -> tuple[float, bool]:
        if not self._in_world_bounds(self._x_m, self._y_m):
            return -float("inf"), True
        c1, c2 = self._circle_centers_m()
        d1 = self._dist_at_m(c1[0], c1[1])
        d2 = self._dist_at_m(c2[0], c2[1])
        r = float(self.footprint.radius_m)
        od_m = min(d1 - r, d2 - r)
        # EDT measures center-to-center distance; add half-cell margin to account
        # for the obstacle cell extending 0.5*cell_size from its center.
        r_col = r + self._half_cell_m
        collision = (d1 <= r_col) or (d2 <= r_col)
        return float(od_m), bool(collision)

    def _od_and_collision_at_pose_m(self, x_m: float, y_m: float, psi_rad: float) -> tuple[float, bool]:
        if not self._in_world_bounds(x_m, y_m):
            return -float("inf"), True

        c = math.cos(float(psi_rad))
        s = math.sin(float(psi_rad))
        c1x = float(x_m) + c * float(self.footprint.x1_m)
        c1y = float(y_m) + s * float(self.footprint.x1_m)
        c2x = float(x_m) + c * float(self.footprint.x2_m)
        c2y = float(y_m) + s * float(self.footprint.x2_m)

        d1 = self._dist_at_m(c1x, c1y)
        d2 = self._dist_at_m(c2x, c2y)
        r = float(self.footprint.radius_m)
        od_m = min(float(d1) - r, float(d2) - r)
        r_col = r + self._half_cell_m
        collision = (float(d1) <= r_col) or (float(d2) <= r_col)
        return float(od_m), bool(collision)

    def _od_and_collision_at_pose_m_vec(
        self,
        x_m: np.ndarray,
        y_m: np.ndarray,
        psi_rad: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        xv = np.asarray(x_m, dtype=np.float64)
        yv = np.asarray(y_m, dtype=np.float64)
        psiv = np.asarray(psi_rad, dtype=np.float64)

        max_x = float(self._width - 1) * float(self.cell_size_m)
        max_y = float(self._height - 1) * float(self.cell_size_m)
        in_bounds = (xv >= 0.0) & (xv <= max_x) & (yv >= 0.0) & (yv <= max_y)

        c = np.cos(psiv)
        s = np.sin(psiv)
        c1x = xv + c * float(self.footprint.x1_m)
        c1y = yv + s * float(self.footprint.x1_m)
        c2x = xv + c * float(self.footprint.x2_m)
        c2y = yv + s * float(self.footprint.x2_m)

        d1 = self._dist_at_m_vec(c1x, c1y)
        d2 = self._dist_at_m_vec(c2x, c2y)
        r = float(self.footprint.radius_m)
        od = np.minimum(d1 - r, d2 - r)
        r_col = r + self._half_cell_m
        coll = (d1 <= r_col) | (d2 <= r_col)

        od = np.where(in_bounds, od, -float("inf")).astype(np.float64, copy=False)
        coll = np.where(in_bounds, coll, True).astype(np.bool_, copy=False)
        return od, coll

    def _cost_to_goal_at_m_vec(self, x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
        xi = np.asarray(x_m, dtype=np.float64) / float(self.cell_size_m)
        yi = np.asarray(y_m, dtype=np.float64) / float(self.cell_size_m)
        return bilinear_sample_2d_finite_vec(
            self._cost_to_goal_m,
            x=xi,
            y=yi,
            fill_value=float(self._cost_fill_m),
        )

    def _cost_to_goal_pose_m_vec(self, x_m: np.ndarray, y_m: np.ndarray, psi_rad: np.ndarray) -> np.ndarray:
        xv = np.asarray(x_m, dtype=np.float64)
        yv = np.asarray(y_m, dtype=np.float64)
        psiv = np.asarray(psi_rad, dtype=np.float64)
        c = np.cos(psiv)
        s = np.sin(psiv)
        c1x = xv + c * float(self.footprint.x1_m)
        c1y = yv + s * float(self.footprint.x1_m)
        c2x = xv + c * float(self.footprint.x2_m)
        c2y = yv + s * float(self.footprint.x2_m)
        c1 = self._cost_to_goal_at_m_vec(c1x, c1y)
        c2 = self._cost_to_goal_at_m_vec(c2x, c2y)
        return np.maximum(c1, c2).astype(np.float64, copy=False)

    def _bicycle_integrate_one_step_vec(
        self,
        *,
        x_m: np.ndarray,
        y_m: np.ndarray,
        psi_rad: np.ndarray,
        v_m_s: np.ndarray,
        delta_rad: np.ndarray,
        delta_dot_rad_s: np.ndarray,
        a_m_s2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dt = float(self.model.dt)
        wheelbase = float(self.model.wheelbase_m)
        v_max = float(self.model.v_max_m_s)
        delta_max = float(self.model.delta_max_rad)

        v_next = np.clip(
            np.asarray(v_m_s, dtype=np.float64) + np.asarray(a_m_s2, dtype=np.float64) * dt,
            -float(v_max),
            float(v_max),
        )

        delta_unclipped = np.asarray(delta_rad, dtype=np.float64) + np.asarray(delta_dot_rad_s, dtype=np.float64) * dt
        delta_next = np.clip(delta_unclipped, -float(delta_max), +float(delta_max))

        psi = np.asarray(psi_rad, dtype=np.float64)
        x_next = np.asarray(x_m, dtype=np.float64) + v_next * np.cos(psi) * dt
        y_next = np.asarray(y_m, dtype=np.float64) + v_next * np.sin(psi) * dt
        psi_next = self._wrap_angle_rad_np(psi + (v_next / wheelbase) * np.tan(delta_next) * dt)
        return x_next, y_next, psi_next, v_next, delta_next

    def _rollout_constant_actions_end_state(
        self,
        *,
        delta_dot_rad_s: np.ndarray,
        a_m_s2: np.ndarray,
        horizon_steps: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        h = max(1, int(horizon_steps))
        delta_dot = np.asarray(delta_dot_rad_s, dtype=np.float64).reshape(-1)
        accel = np.asarray(a_m_s2, dtype=np.float64).reshape(-1)
        if delta_dot.shape != accel.shape:
            raise ValueError("delta_dot_rad_s and a_m_s2 must have the same shape")

        n = int(delta_dot.size)
        x = np.full((n,), float(self._x_m), dtype=np.float64)
        y = np.full((n,), float(self._y_m), dtype=np.float64)
        psi = np.full((n,), float(self._psi_rad), dtype=np.float64)
        v = np.full((n,), float(self._v_m_s), dtype=np.float64)
        delta = np.full((n,), float(self._delta_rad), dtype=np.float64)

        min_od = np.full((n,), float("inf"), dtype=np.float64)
        coll = np.zeros((n,), dtype=np.bool_)
        reached = np.zeros((n,), dtype=np.bool_)
        active = np.ones((n,), dtype=np.bool_)

        gx_m = float(self.goal_xy[0]) * float(self.cell_size_m)
        gy_m = float(self.goal_xy[1]) * float(self.cell_size_m)
        tol_m = float(self.goal_tolerance_m)

        for _ in range(h):
            if not bool(active.any()):
                break

            x1, y1, psi1, v1, delta1 = self._bicycle_integrate_one_step_vec(
                x_m=x,
                y_m=y,
                psi_rad=psi,
                v_m_s=v,
                delta_rad=delta,
                delta_dot_rad_s=delta_dot,
                a_m_s2=accel,
            )
            # Freeze terminated rollouts (reached/collided) so later steps do not affect masks.
            x = np.where(active, x1, x)
            y = np.where(active, y1, y)
            psi = np.where(active, psi1, psi)
            v = np.where(active, v1, v)
            delta = np.where(active, delta1, delta)

            od, coll_step = self._od_and_collision_at_pose_m_vec(x, y, psi)
            min_od = np.where(active, np.minimum(min_od, od), min_od)
            coll_now = coll_step & active
            coll |= coll_now

            d_goal_m = np.hypot(float(gx_m) - x, float(gy_m) - y)
            reached_now = (d_goal_m <= float(tol_m)) & active & (~coll_now)
            reached |= reached_now

            active &= ~(coll_now | reached_now)

        return x, y, psi, v, min_od, coll, reached

    def _fallback_action_short_rollout(
        self,
        *,
        horizon_steps: int,
        min_od_m: float = 0.0,
    ) -> int:
        """Fallback action chooser when Hybrid A* guidance is unavailable.

        Chooses the collision-free action that yields the lowest cost-to-go after a short
        constant-action rollout. This is used only as a last resort to keep rollouts moving.
        """

        h = max(1, int(horizon_steps))
        min_od_thr = float(min_od_m)
        delta_dot = self.action_table[:, 0]
        accel = self.action_table[:, 1]
        x, y, psi, _v, min_od, coll, reached = self._rollout_constant_actions_end_state(
            delta_dot_rad_s=delta_dot,
            a_m_s2=accel,
            horizon_steps=h,
        )
        cost1 = self._cost_to_goal_pose_m_vec(x, y, psi)

        ok = (~coll) & (min_od >= float(min_od_thr)) & np.isfinite(cost1)
        if bool(ok.any()):
            ok_reached = ok & reached
            idx = np.nonzero(ok_reached if bool(ok_reached.any()) else ok)[0]
            costs = cost1[idx]
            ods = min_od[idx]
            best_cost = float(np.min(costs))
            cand = idx[costs <= float(best_cost) + 1e-9]
            if cand.size == 0:
                cand = idx[int(np.argmin(costs))]
                return int(cand)
            best = int(cand[int(np.argmax(min_od[cand]))])
            return int(best)

        # Last resort: pick the one-step action with maximum clearance (even if it still collides).
        x0 = float(self._x_m)
        y0 = float(self._y_m)
        psi0 = float(self._psi_rad)
        v0 = float(self._v_m_s)
        delta0 = float(self._delta_rad)

        best_action = 0
        best_od = -float("inf")
        for a_id in range(int(self.action_table.shape[0])):
            delta_dot = float(self.action_table[a_id, 0])
            a = float(self.action_table[a_id, 1])
            x, y, psi, _v, _delta = bicycle_integrate_one_step(
                x_m=x0,
                y_m=y0,
                psi_rad=psi0,
                v_m_s=v0,
                delta_rad=delta0,
                delta_dot_rad_s=delta_dot,
                a_m_s2=a,
                params=self.model,
            )
            od, _coll = self._od_and_collision_at_pose_m(x, y, psi)
            if float(od) > float(best_od):
                best_od = float(od)
                best_action = int(a_id)
        return int(best_action)

    def _hybrid_astar_path(self, *, start_xy: tuple[int, int], timeout_s: float = 5.0, max_nodes: int = 200_000) -> list[tuple[float, float]]:
        key = (int(start_xy[0]), int(start_xy[1]), int(self.goal_xy[0]), int(self.goal_xy[1]))
        cached = self._ha_path_cache.get(key)
        if cached is not None:
            return cached

        # Fast path: load precomputed Hybrid A* reference paths for the canonical forest starts.
        # These are deterministic given the fixed forest seeds and avoid paying planning cost during training/inference.
        if (
            self.map_spec.name.startswith("forest_")
            and (int(start_xy[0]), int(start_xy[1])) == (int(self._canonical_start_xy[0]), int(self._canonical_start_xy[1]))
            and (int(self.goal_xy[0]), int(self.goal_xy[1])) == (int(self._canonical_goal_xy[0]), int(self._canonical_goal_xy[1]))
        ):
            pre = Path(__file__).resolve().parent / "maps" / "precomputed" / f"{self.map_spec.name}_hybrid_astar_path.json"
            if pre.exists():
                try:
                    payload = json.loads(pre.read_text(encoding="utf-8"))
                    pts = payload.get("path_xy_cells")
                    if bool(payload.get("success")) and isinstance(pts, list) and len(pts) >= 2:
                        path = [(float(p[0]), float(p[1])) for p in pts]
                        h, w = self._grid.shape
                        sx, sy = int(self._canonical_start_xy[0]), int(self._canonical_start_xy[1])
                        gx, gy = int(self._canonical_goal_xy[0]), int(self._canonical_goal_xy[1])
                        tol_cells = float(self.goal_tolerance_m) / float(self.cell_size_m)

                        def cell_free(xc: float, yc: float) -> bool:
                            xi = int(round(float(xc)))
                            yi = int(round(float(yc)))
                            if not (0 <= xi < w and 0 <= yi < h):
                                return False
                            return int(self._grid[yi, xi]) == 0

                        start_ok = (float(path[0][0]) - float(sx)) ** 2 + (float(path[0][1]) - float(sy)) ** 2 <= 4.0
                        goal_ok = (float(path[-1][0]) - float(gx)) ** 2 + (float(path[-1][1]) - float(gy)) ** 2 <= (tol_cells + 2.0) ** 2
                        path_ok = bool(start_ok) and bool(goal_ok) and all(cell_free(x, y) for x, y in path)
                        if path_ok:
                            self._ha_path_cache[key] = path
                            return path
                except Exception:
                    pass

        try:
            from amr_dqn.baselines.pathplan import (
                default_ackermann_params,
                forest_two_circle_footprint,
                grid_map_from_obstacles,
                plan_hybrid_astar,
            )
        except Exception:
            self._ha_path_cache[key] = []
            return []

        grid_map = grid_map_from_obstacles(grid_y0_bottom=self._grid, cell_size_m=float(self.cell_size_m))
        params = default_ackermann_params(
            wheelbase_m=float(self.model.wheelbase_m),
            delta_max_rad=float(self.model.delta_max_rad),
            v_max_m_s=float(self.model.v_max_m_s),
        )
        footprint = forest_two_circle_footprint()

        res = plan_hybrid_astar(
            grid_map=grid_map,
            footprint=footprint,
            params=params,
            start_xy=(int(start_xy[0]), int(start_xy[1])),
            goal_xy=(int(self.goal_xy[0]), int(self.goal_xy[1])),
            goal_theta_rad=0.0,
            start_theta_rad=None,
            goal_xy_tol_m=float(self.goal_tolerance_m),
            goal_theta_tol_rad=float(math.pi),
            timeout_s=float(timeout_s),
            max_nodes=int(max_nodes),
        )
        path = list(res.path_xy_cells) if res.success else []
        self._ha_path_cache[key] = path
        return path

    def expert_action_hybrid_astar(
        self,
        *,
        lookahead_points: int = 3,
        horizon_steps: int = 4,
        w_target: float = 0.5,
        w_heading: float = 0.4,
        w_clearance: float = 0.2,
        w_speed: float = 0.0,
    ) -> int:
        """Hybrid-A* guided expert (DQfD demos / guided exploration).

        Computes a Hybrid A* reference path once per episode-start (cached by start cell) and then
        tracks it with a pure-pursuit style steering target + discrete control selection under a
        short-horizon safety mask.
        """

        path = self._hybrid_astar_path(start_xy=self._ha_start_xy)
        if len(path) < 2:
            return self._fallback_action_short_rollout(horizon_steps=int(horizon_steps), min_od_m=0.0)

        x_cells = float(self._x_m) / float(self.cell_size_m)
        y_cells = float(self._y_m) / float(self.cell_size_m)

        # Find nearest path index (limited window around previous index).
        start_i = max(0, int(self._ha_progress_idx) - 25)
        end_i = min(len(path), int(self._ha_progress_idx) + 250)
        if end_i <= start_i:
            start_i, end_i = 0, len(path)
        best_i = start_i
        best_d2 = float("inf")
        for i in range(start_i, end_i):
            px, py = path[i]
            d2 = (float(px) - x_cells) ** 2 + (float(py) - y_cells) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        self._ha_progress_idx = int(best_i)

        la = max(1, int(lookahead_points))
        tgt_i = min(int(best_i) + la, len(path) - 1)
        tx_cells, ty_cells = path[tgt_i]
        tx_m = float(tx_cells) * float(self.cell_size_m)
        ty_m = float(ty_cells) * float(self.cell_size_m)

        h = max(1, int(horizon_steps))
        delta_dot = self.action_table[:, 0]
        accel = self.action_table[:, 1]
        x, y, psi, v, min_od, coll, _reached = self._rollout_constant_actions_end_state(
            delta_dot_rad_s=delta_dot,
            a_m_s2=accel,
            horizon_steps=h,
        )

        cost = self._cost_to_goal_pose_m_vec(x, y, psi)
        dist_tgt = np.hypot(float(tx_m) - x, float(ty_m) - y)
        tgt_heading = np.arctan2(float(ty_m) - y, float(tx_m) - x)
        heading_err = self._wrap_angle_rad_np(tgt_heading - psi)

        score = -cost
        score += -float(w_target) * dist_tgt - float(w_heading) * np.abs(heading_err)
        score += float(w_clearance) * min_od

        if float(w_speed) != 0.0:
            v_max = float(self.model.v_max_m_s)
            score += float(w_speed) * (v / max(1e-9, float(v_max)))

        invalid = coll | (~np.isfinite(score)) | (~np.isfinite(cost))
        score = np.where(invalid, -float("inf"), score)
        best_action = int(np.argmax(score))
        if not math.isfinite(float(score[best_action])):
            return self._fallback_action_short_rollout(horizon_steps=int(horizon_steps), min_od_m=0.0)

        return int(best_action)

    def expert_action_cost_to_go(
        self,
        *,
        horizon_steps: int = 15,
        min_od_m: float = 0.0,
    ) -> int:
        """Lightweight expert based on short-horizon rollouts over the clearance-aware cost-to-go field."""
        return self._fallback_action_short_rollout(horizon_steps=int(horizon_steps), min_od_m=float(min_od_m))

    def _rollout_constant_action_metrics(
        self,
        a_id: int,
        *,
        horizon_steps: int,
    ) -> tuple[float, float, float, bool, bool]:
        """Simulate a constant discrete action for a short horizon.

        Returns: (cost_to_go_end, v_end, min_od_over_horizon, collision_over_horizon, reached_within_horizon).
        """

        h = max(1, int(horizon_steps))
        a_id = int(a_id)

        delta_dot = float(self.action_table[a_id, 0])
        a = float(self.action_table[a_id, 1])

        x = float(self._x_m)
        y = float(self._y_m)
        psi = float(self._psi_rad)
        v = float(self._v_m_s)
        delta = float(self._delta_rad)

        gx_m = float(self.goal_xy[0]) * float(self.cell_size_m)
        gy_m = float(self.goal_xy[1]) * float(self.cell_size_m)
        tol_m = float(self.goal_tolerance_m)

        min_od = float("inf")
        for _ in range(h):
            x, y, psi, v, delta = bicycle_integrate_one_step(
                x_m=x,
                y_m=y,
                psi_rad=psi,
                v_m_s=v,
                delta_rad=delta,
                delta_dot_rad_s=delta_dot,
                a_m_s2=a,
                params=self.model,
            )
            od, coll = self._od_and_collision_at_pose_m(x, y, psi)
            min_od = min(float(min_od), float(od))
            if coll:
                return float("inf"), float(v), float(min_od), True, False

            if float(math.hypot(float(gx_m) - float(x), float(gy_m) - float(y))) <= float(tol_m):
                cost = float(self._cost_to_goal_pose_m(x, y, psi))
                return float(cost), float(v), float(min_od), False, True

        cost = float(self._cost_to_goal_pose_m(x, y, psi))
        return float(cost), float(v), float(min_od), False, False

    def is_action_safe(
        self,
        a_id: int,
        *,
        horizon_steps: int = 10,
        min_od_m: float = 0.0,
    ) -> bool:
        _cost, _v, min_od, coll, _reached = self._rollout_constant_action_metrics(
            int(a_id), horizon_steps=int(horizon_steps)
        )
        if bool(coll):
            return False
        return float(min_od) >= float(min_od_m)

    def is_action_admissible(
        self,
        a_id: int,
        *,
        horizon_steps: int = 10,
        min_od_m: float = 0.0,
        min_progress_m: float = 1e-4,
        allow_reverse: bool = True,
    ) -> bool:
        cost0 = float(self._cost_to_goal_pose_m(float(self._x_m), float(self._y_m), float(self._psi_rad)))
        if not math.isfinite(cost0):
            return True

        # Progress is judged at the end of the short-horizon constant-action rollout, while safety
        # (collision / clearance) is judged over the same horizon.
        h = max(1, int(horizon_steps))
        cost1, v_end, min_od, coll, reached = self._rollout_constant_action_metrics(int(a_id), horizon_steps=h)
        if bool(coll):
            return False
        if float(min_od) < float(min_od_m):
            return False
        if bool(reached):
            return True
        if not math.isfinite(cost1):
            return False
        if float(cost0 - cost1) >= float(min_progress_m):
            return True

        # Allow backing up / reversing only when no forward-progress actions exist under the same
        # short-horizon constraints. This avoids the degenerate near-goal behavior where the policy
        # keeps selecting reverse/stop-like actions and triggers stuck termination.
        if bool(allow_reverse):
            reverse_v_min = 0.10
            if float(v_end) < -float(reverse_v_min):
                prog_mask = self.admissible_action_mask(
                    horizon_steps=h,
                    min_od_m=float(min_od_m),
                    min_progress_m=float(min_progress_m),
                    fallback_to_safe=False,
                    allow_reverse=False,
                )
                if not bool(prog_mask.any()):
                    return True
        return False

    def safe_action_mask(
        self,
        *,
        horizon_steps: int = 10,
        min_od_m: float = 0.0,
    ) -> np.ndarray:
        """Return a boolean mask of actions that remain collision-free over a short horizon."""
        h = max(1, int(horizon_steps))
        min_od_thr = float(min_od_m)
        delta_dot = self.action_table[:, 0]
        accel = self.action_table[:, 1]
        _x, _y, _psi, _v, min_od, coll, _reached = self._rollout_constant_actions_end_state(
            delta_dot_rad_s=delta_dot,
            a_m_s2=accel,
            horizon_steps=h,
        )
        out = (~coll) & (min_od >= float(min_od_thr))
        return out.astype(np.bool_, copy=False)

    def admissible_action_mask(
        self,
        *,
        horizon_steps: int = 10,
        min_od_m: float = 0.0,
        min_progress_m: float = 1e-4,
        fallback_to_safe: bool = True,
        allow_reverse: bool = True,
    ) -> np.ndarray:
        """Mask actions that are safe and make cost-to-go progress (optionally allow reverse)."""

        cost0 = float(self._cost_to_goal_pose_m(float(self._x_m), float(self._y_m), float(self._psi_rad)))
        out = np.zeros((int(self.action_table.shape[0]),), dtype=np.bool_)
        if not math.isfinite(cost0):
            out[:] = True
            return out

        h = max(1, int(horizon_steps))
        min_od_thr = float(min_od_m)
        min_prog = float(min_progress_m)

        delta_dot = self.action_table[:, 0]
        accel = self.action_table[:, 1]
        x, y, psi, v_end, min_od, coll, reached = self._rollout_constant_actions_end_state(
            delta_dot_rad_s=delta_dot,
            a_m_s2=accel,
            horizon_steps=h,
        )
        cost1 = self._cost_to_goal_pose_m_vec(x, y, psi)

        safe = (~coll) & (min_od >= float(min_od_thr)) & np.isfinite(cost1)
        prog = ((float(cost0) - cost1) >= float(min_prog)) | reached
        out = safe & prog
        if bool(allow_reverse) and not bool(out.any()):
            # Only expose reverse actions when no progress actions exist.
            reverse_v_min = 0.10
            out = safe & (v_end < -float(reverse_v_min))

        # Fallback: if everything is filtered out, keep the collision-safe actions.
        if bool(fallback_to_safe) and not bool(out.any()):
            out = (~coll) & (min_od >= float(min_od_thr))
        return out.astype(np.bool_, copy=False)

    def _observe(self) -> np.ndarray:
        # Normalized (x,y) + goal (x,y) in [-1,1].
        max_x = max(1e-6, float(self._width - 1) * self.cell_size_m)
        max_y = max(1e-6, float(self._height - 1) * self.cell_size_m)
        ax_n = 2.0 * (float(self._x_m) / float(max_x)) - 1.0
        ay_n = 2.0 * (float(self._y_m) / float(max_y)) - 1.0
        gx_n = 2.0 * ((float(self.goal_xy[0]) * self.cell_size_m) / float(max_x)) - 1.0
        gy_n = 2.0 * ((float(self.goal_xy[1]) * self.cell_size_m) / float(max_y)) - 1.0

        # Scalars
        sin_psi = float(math.sin(float(self._psi_rad)))
        cos_psi = float(math.cos(float(self._psi_rad)))
        v_n = float(self._v_m_s) / float(self.model.v_max_m_s)
        delta_lim = float(self.model.delta_max_rad)
        delta_n = 0.0 if abs(delta_lim) < 1e-9 else float(self._delta_rad) / float(delta_lim)
        cost = self._cost_to_goal_pose_m(self._x_m, self._y_m, float(self._psi_rad))
        if math.isfinite(cost):
            cost01 = float(cost) / max(1e-6, float(self._cost_norm_m))
        else:
            cost01 = float(self._distance_to_goal_m()) / max(1e-6, float(self._diag_m))
        cost_n = 2.0 * float(np.clip(cost01, 0.0, 1.0)) - 1.0

        alpha_n = float(self._goal_relative_angle_rad()) / math.pi
        od01 = min(self.od_cap_m, max(0.0, float(self._last_od_m))) / float(self.od_cap_m)
        od_n = 2.0 * float(np.clip(od01, 0.0, 1.0)) - 1.0

        # Clamp to stable ranges.
        ax_n = float(np.clip(ax_n, -1.0, 1.0))
        ay_n = float(np.clip(ay_n, -1.0, 1.0))
        gx_n = float(np.clip(gx_n, -1.0, 1.0))
        gy_n = float(np.clip(gy_n, -1.0, 1.0))
        sin_psi = float(np.clip(sin_psi, -1.0, 1.0))
        cos_psi = float(np.clip(cos_psi, -1.0, 1.0))
        v_n = float(np.clip(v_n, -1.0, 1.0))
        delta_n = float(np.clip(delta_n, -1.0, 1.0))
        cost_n = float(np.clip(cost_n, -1.0, 1.0))
        alpha_n = float(np.clip(alpha_n, -1.0, 1.0))
        obs = np.concatenate(
            [
                np.array(
                    [ax_n, ay_n, gx_n, gy_n, sin_psi, cos_psi, v_n, delta_n, cost_n, alpha_n, od_n],
                    dtype=np.float32,
                ),
                self._obs_occ_flat,
                self._obs_cost_flat,
            ]
        )
        return obs.astype(np.float32, copy=False)

    def _distance_to_goal_m(self) -> float:
        gx = float(self.goal_xy[0]) * self.cell_size_m
        gy = float(self.goal_xy[1]) * self.cell_size_m
        return float(math.hypot(gx - float(self._x_m), gy - float(self._y_m)))

    def _goal_relative_angle_rad(self) -> float:
        gx = float(self.goal_xy[0]) * self.cell_size_m
        gy = float(self.goal_xy[1]) * self.cell_size_m
        goal_heading = math.atan2(gy - float(self._y_m), gx - float(self._x_m))
        return wrap_angle_rad(float(goal_heading) - float(self._psi_rad))

    def _cost_to_goal_at_m(self, x_m: float, y_m: float) -> float:
        xi = float(x_m) / self.cell_size_m
        yi = float(y_m) / self.cell_size_m
        return bilinear_sample_2d_finite(self._cost_to_goal_m, x=xi, y=yi, fill_value=float(self._cost_fill_m))

    def _cost_to_goal_pose_m(self, x_m: float, y_m: float, psi_rad: float) -> float:
        """Cost-to-go for the whole vehicle footprint (two circles).

        Uses the max over circle-center costs so progress shaping does not
        encourage motions where one circle becomes trapped/unsafe.
        """
        c = math.cos(float(psi_rad))
        s = math.sin(float(psi_rad))
        c1x = float(x_m) + c * float(self.footprint.x1_m)
        c1y = float(y_m) + s * float(self.footprint.x1_m)
        c2x = float(x_m) + c * float(self.footprint.x2_m)
        c2y = float(y_m) + s * float(self.footprint.x2_m)
        c1 = self._cost_to_goal_at_m(c1x, c1y)
        c2 = self._cost_to_goal_at_m(c2x, c2y)
        return float(max(float(c1), float(c2)))

    def _in_world_bounds(self, x_m: float, y_m: float) -> bool:
        max_x = float(self._width - 1) * self.cell_size_m
        max_y = float(self._height - 1) * self.cell_size_m
        return (0.0 <= float(x_m) <= max_x) and (0.0 <= float(y_m) <= max_y)

    def _sector_ray_distances_n(self) -> np.ndarray:
        """LIDAR-like ray distances, normalized to [0,1]. Angles are in the vehicle frame."""
        x0 = float(self._x_m) / self.cell_size_m
        y0 = float(self._y_m) / self.cell_size_m
        max_range_cells = float(self.sensor_range_m) / self.cell_size_m
        step_cells = 0.5
        max_steps = max(1, int(math.ceil(max_range_cells / step_cells)))

        out = np.ones((self.n_sectors,), dtype=np.float32)
        for i in range(self.n_sectors):
            ang = float(self._psi_rad) + (2.0 * math.pi) * (float(i) / float(self.n_sectors))
            c = math.cos(ang)
            s = math.sin(ang)
            hit_cells = max_range_cells
            for j in range(1, max_steps + 1):
                d_cells = float(j) * step_cells
                xi = x0 + c * d_cells
                yi = y0 + s * d_cells
                ix = int(math.floor(xi + 0.5))
                iy = int(math.floor(yi + 0.5))
                if not (0 <= ix < self._width and 0 <= iy < self._height):
                    hit_cells = d_cells
                    break
                if self._grid[iy, ix] == 1:
                    hit_cells = d_cells
                    break
            out[i] = float(np.clip((hit_cells * self.cell_size_m) / float(self.sensor_range_m), 0.0, 1.0))
        return out
