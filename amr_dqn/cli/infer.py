from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from amr_dqn.config_io import apply_config_defaults, load_json, resolve_config_path, select_section
from amr_dqn.runtime import configure_runtime, select_device, torch_runtime_info
from amr_dqn.runs import create_run_dir, resolve_experiment_dir, resolve_models_dir

configure_runtime()

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import gymnasium as gym
import numpy as np
import pandas as pd
import torch

from amr_dqn.agents import AgentConfig, DQNFamilyAgent, parse_rl_algo
from amr_dqn.baselines.pathplan import (
    default_ackermann_params,
    PlannerResult,
    forest_two_circle_footprint,
    grid_map_from_obstacles,
    plan_hybrid_astar,
    plan_rrt_star,
    point_footprint,
)
from amr_dqn.env import AMRBicycleEnv, AMRGridEnv, RewardWeights
from amr_dqn.forest_policy import forest_select_action
from amr_dqn.maps import FOREST_ENV_ORDER, get_map_spec
from amr_dqn.metrics import KPI, avg_abs_curvature, max_corner_degree, num_path_corners, path_length
from amr_dqn.smoothing import chaikin_smooth


def _safe_slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")


@dataclass(frozen=True)
class PathTrace:
    path_xy_cells: list[tuple[float, float]]
    success: bool


@dataclass(frozen=True)
class ControlTrace:
    t_s: np.ndarray
    v_m_s: np.ndarray
    delta_rad: np.ndarray


@dataclass(frozen=True)
class RolloutResult:
    path_xy_cells: list[tuple[float, float]]
    compute_time_s: float
    reached: bool
    steps: int
    path_time_s: float
    controls: ControlTrace | None = None


def _env_dt_s(env: gym.Env) -> float:
    if isinstance(env, AMRBicycleEnv):
        return float(env.model.dt)
    return 1.0


def rollout_agent(
    env: gym.Env,
    agent: DQNFamilyAgent,
    *,
    max_steps: int,
    seed: int,
    reset_options: dict[str, object] | None = None,
    time_mode: str = "rollout",
    obs_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    forest_adm_horizon: int = 15,
    forest_topk: int = 10,
    forest_min_od_m: float = 0.0,
    forest_min_progress_m: float = 1e-4,
    collect_controls: bool = False,
) -> RolloutResult:
    obs, _info0 = env.reset(seed=seed, options=reset_options)
    if obs_transform is not None:
        obs = obs_transform(obs)
    path: list[tuple[float, float]] = [(float(env.start_xy[0]), float(env.start_xy[1]))]
    dt_s = float(_env_dt_s(env))

    t_series: list[float] | None = None
    v_series: list[float] | None = None
    delta_series: list[float] | None = None
    if bool(collect_controls) and isinstance(env, AMRBicycleEnv):
        t_series = [0.0]
        v_series = [float(getattr(env, "_v_m_s", 0.0))]
        delta_series = [float(getattr(env, "_delta_rad", 0.0))]

    def sync_cuda() -> None:
        if agent.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

    time_mode = str(time_mode).lower().strip()
    if time_mode not in {"rollout", "policy"}:
        raise ValueError("time_mode must be one of: rollout, policy")

    inference_time_s = 0.0
    sync_cuda()
    t_rollout0 = time.perf_counter()
    done = False
    truncated = False
    steps = 0
    reached = False
    adm_h = max(1, int(forest_adm_horizon))
    topk_k = max(1, int(forest_topk))
    min_od = float(forest_min_od_m)
    min_prog = float(forest_min_progress_m)

    while not (done or truncated) and steps < max_steps:
        steps += 1
        if time_mode == "policy":
            sync_cuda()
            t0 = time.perf_counter()
        if isinstance(env, AMRBicycleEnv):
            a = forest_select_action(
                env, agent, obs,
                episode=0, explore=False,
                horizon_steps=adm_h, topk=topk_k,
                min_od_m=min_od, min_progress_m=min_prog,
            )
        else:
            a = agent.act(obs, episode=0, explore=False)
        if time_mode == "policy":
            sync_cuda()
            inference_time_s += float(time.perf_counter() - t0)
        obs, _, done, truncated, info = env.step(a)
        if obs_transform is not None:
            obs = obs_transform(obs)
        x, y = info["agent_xy"]
        path.append((float(x), float(y)))
        if t_series is not None and v_series is not None and delta_series is not None:
            t_series.append(float(steps) * dt_s)
            v_series.append(float(info.get("v_m_s", float(getattr(env, "_v_m_s", 0.0)))))
            delta_series.append(float(info.get("delta_rad", float(getattr(env, "_delta_rad", 0.0)))))
        if info.get("reached"):
            reached = True
            break

    if time_mode == "rollout":
        sync_cuda()
        inference_time_s = float(time.perf_counter() - t_rollout0)
    controls = None
    if t_series is not None and v_series is not None and delta_series is not None:
        controls = ControlTrace(
            t_s=np.asarray(t_series, dtype=np.float64),
            v_m_s=np.asarray(v_series, dtype=np.float64),
            delta_rad=np.asarray(delta_series, dtype=np.float64),
        )
    return RolloutResult(
        path_xy_cells=path,
        compute_time_s=float(inference_time_s),
        reached=bool(reached),
        steps=int(steps),
        path_time_s=float(steps) * dt_s,
        controls=controls,
    )


def rollout_tracked_path_mpc(
    env: AMRBicycleEnv,
    ref_path_xy_cells: list[tuple[float, float]],
    *,
    max_steps: int,
    seed: int,
    reset_options: dict[str, object] | None = None,
    time_mode: str = "rollout",
    trace_path: Path | None = None,
    lookahead_points: int = 5,
    horizon_steps: int = 15,
    n_candidates: int = 512,
    w_target: float = 0.2,
    w_heading: float = 0.2,
    w_clearance: float = 0.8,
    w_speed: float = 0.0,
    w_control: float = 0.01,
    collect_controls: bool = False,
) -> RolloutResult:
    """Continuous-control MPC-style tracker for baseline paths (forest only).

    This tracker does NOT use the discrete `action_table`. Instead, it samples continuous control
    candidates `(delta_dot, a)`, evaluates them with a short horizon rollout, and applies the best
    control using `AMRBicycleEnv.step_continuous(...)`.
    """
    time_mode = str(time_mode).lower().strip()
    if time_mode not in {"rollout", "policy"}:
        raise ValueError("time_mode must be one of: rollout, policy")

    obs, _info0 = env.reset(seed=seed, options=reset_options)
    path: list[tuple[float, float]] = [(float(env.start_xy[0]), float(env.start_xy[1]))]
    dt_s = float(_env_dt_s(env))

    t_series: list[float] | None = None
    v_series: list[float] | None = None
    delta_series: list[float] | None = None
    if bool(collect_controls):
        t_series = [0.0]
        v_series = [float(getattr(env, "_v_m_s", 0.0))]
        delta_series = [float(getattr(env, "_delta_rad", 0.0))]
    trace_rows: list[dict[str, object]] | None = [] if trace_path is not None else None
    if trace_rows is not None:
        d_goal0 = float(env._distance_to_goal_m())
        alpha0 = float(env._goal_relative_angle_rad())
        reached0 = (d_goal0 <= float(env.goal_tolerance_m)) and (abs(alpha0) <= float(env.goal_angle_tolerance_rad))
        trace_rows.append(
            {
                "step": 0,
                "x_m": float(env._x_m),
                "y_m": float(env._y_m),
                "theta_rad": float(env._psi_rad),
                "v_m_s": float(env._v_m_s),
                "delta_rad": float(env._delta_rad),
                "delta_dot_rad_s": 0.0,
                "a_m_s2": 0.0,
                "od_m": float(getattr(env, "_last_od_m", 0.0)),
                "collision": bool(getattr(env, "_last_collision", False)),
                "reached": bool(reached0),
                "stuck": False,
            }
        )
    if len(ref_path_xy_cells) < 2:
        controls = None
        if t_series is not None and v_series is not None and delta_series is not None:
            controls = ControlTrace(
                t_s=np.asarray(t_series, dtype=np.float64),
                v_m_s=np.asarray(v_series, dtype=np.float64),
                delta_rad=np.asarray(delta_series, dtype=np.float64),
            )
        return RolloutResult(
            path_xy_cells=path,
            compute_time_s=0.0,
            reached=False,
            steps=0,
            path_time_s=0.0,
            controls=controls,
        )

    # Precompute reference arc-length (meters) for progress-based tracking.
    ref_xy = np.asarray(ref_path_xy_cells, dtype=np.float64)
    if ref_xy.shape[0] >= 2:
        d = np.diff(ref_xy, axis=0)
        ds = np.hypot(d[:, 0], d[:, 1]) * float(env.cell_size_m)
        ref_s_m = np.concatenate([np.array([0.0], dtype=np.float64), np.cumsum(ds, dtype=np.float64)], axis=0)
    else:
        ref_s_m = np.zeros((ref_xy.shape[0],), dtype=np.float64)

    n = max(16, int(n_candidates))
    h = max(1, int(horizon_steps))
    la = max(1, int(lookahead_points))
    v_max = float(env.model.v_max_m_s)
    dd_max = float(env.model.delta_dot_max_rad_s)
    a_max = float(env.model.a_max_m_s2)
    rng = getattr(env, "_rng", np.random.default_rng(int(seed)))

    def choose_controls(progress_idx: int) -> tuple[float, float, int]:
        x_cells = float(env._x_m) / float(env.cell_size_m)
        y_cells = float(env._y_m) / float(env.cell_size_m)

        # Find nearest reference-path index (windowed search around previous index).
        start_i = max(0, int(progress_idx) - 25)
        end_i = min(len(ref_path_xy_cells), int(progress_idx) + 250)
        if end_i <= start_i:
            start_i, end_i = 0, len(ref_path_xy_cells)
        best_i = start_i
        best_d2 = float("inf")
        for i in range(start_i, end_i):
            px, py = ref_path_xy_cells[i]
            d2 = (float(px) - x_cells) ** 2 + (float(py) - y_cells) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        # Monotonic progress avoids getting "stuck" on self-intersections / loops.
        progress_idx = max(int(progress_idx), int(best_i))

        tgt_i = min(int(progress_idx) + la, len(ref_path_xy_cells) - 1)
        tx_cells, ty_cells = ref_path_xy_cells[tgt_i]
        tx_m = float(tx_cells) * float(env.cell_size_m)
        ty_m = float(ty_cells) * float(env.cell_size_m)

        # Candidate continuous controls.
        delta_dot = rng.uniform(-dd_max, +dd_max, size=(n,)).astype(np.float64, copy=False)
        accel = rng.uniform(-a_max, +a_max, size=(n,)).astype(np.float64, copy=False)

        # Deterministic anchors (help stability / reproducibility).
        anchors = np.array(
            [
                (0.0, 0.0),
                (0.0, +a_max),
                (0.0, -a_max),
                (+dd_max, 0.0),
                (-dd_max, 0.0),
            ],
            dtype=np.float64,
        )
        delta_dot[: anchors.shape[0]] = anchors[:, 0]
        accel[: anchors.shape[0]] = anchors[:, 1]

        # Evaluate candidates with a constant-control horizon rollout (vectorized in the env).
        x, y, psi, v, min_od, coll, reached = env._rollout_constant_actions_end_state(
            delta_dot_rad_s=delta_dot,
            a_m_s2=accel,
            horizon_steps=h,
        )
        cost1 = env._cost_to_goal_pose_m_vec(x, y, psi)
        dist_tgt = np.hypot(float(tx_m) - x, float(ty_m) - y)
        tgt_heading = np.arctan2(float(ty_m) - y, float(tx_m) - x)
        heading_err = env._wrap_angle_rad_np(tgt_heading - psi)

        # Score (higher is better).
        score = -cost1
        score += -float(w_target) * dist_tgt - float(w_heading) * np.abs(heading_err)
        score += float(w_clearance) * min_od
        # Progress reward: meters advanced along the reference arc-length.
        if int(progress_idx) < int(ref_s_m.shape[0]):
            start_s = float(ref_s_m[int(progress_idx)])
            proj_start = int(progress_idx)
            proj_end = min(len(ref_path_xy_cells), int(progress_idx) + 250)
            if proj_end <= proj_start:
                proj_start, proj_end = 0, len(ref_path_xy_cells)
            window = ref_xy[int(proj_start) : int(proj_end)]
            if window.shape[0] > 0:
                x_pred_cells = x / float(env.cell_size_m)
                y_pred_cells = y / float(env.cell_size_m)
                dx = window[:, 0][None, :] - x_pred_cells[:, None]
                dy = window[:, 1][None, :] - y_pred_cells[:, None]
                nearest = np.argmin(dx * dx + dy * dy, axis=1)
                nearest_idx = (int(proj_start) + nearest.astype(np.int32, copy=False)).astype(np.int32, copy=False)
                nearest_idx = np.clip(nearest_idx, 0, int(ref_s_m.shape[0]) - 1)
                progress_m = np.maximum(0.0, ref_s_m[nearest_idx] - float(start_s))
                score += 1.0 * progress_m
        if float(w_speed) != 0.0 and float(v_max) > 1e-6:
            score += float(w_speed) * (v / float(v_max))
        if float(w_control) != 0.0:
            dd_n = delta_dot / max(1e-9, float(dd_max))
            a_n = accel / max(1e-9, float(a_max))
            score -= float(w_control) * (dd_n * dd_n + a_n * a_n)

        ok = (~coll) & np.isfinite(cost1)
        ok_reached = ok & reached
        if bool(ok_reached.any()):
            idx = np.nonzero(ok_reached)[0]
            best = int(idx[int(np.argmax(score[idx]))])
            return float(delta_dot[best]), float(accel[best]), int(progress_idx)
        if bool(ok.any()):
            idx = np.nonzero(ok)[0]
            best = int(idx[int(np.argmax(score[idx]))])
            return float(delta_dot[best]), float(accel[best]), int(progress_idx)

        # Fallback: pick the candidate that maximizes clearance, even if it looks bad.
        best = int(np.argmax(min_od))
        return float(delta_dot[best]), float(accel[best]), int(progress_idx)

    inference_time_s = 0.0
    t_rollout0 = time.perf_counter()
    done = False
    truncated = False
    steps = 0
    reached = False
    progress_idx = 0
    while not (done or truncated) and steps < max_steps:
        steps += 1
        t0 = time.perf_counter() if time_mode == "policy" else None
        delta_dot, accel, progress_idx = choose_controls(progress_idx)
        if t0 is not None:
            inference_time_s += float(time.perf_counter() - t0)
        obs, _, done, truncated, info = env.step_continuous(delta_dot_rad_s=float(delta_dot), a_m_s2=float(accel))
        x, y = info["agent_xy"]
        path.append((float(x), float(y)))
        if t_series is not None and v_series is not None and delta_series is not None:
            t_series.append(float(steps) * dt_s)
            v_series.append(float(info.get("v_m_s", float(getattr(env, "_v_m_s", 0.0)))))
            delta_series.append(float(info.get("delta_rad", float(getattr(env, "_delta_rad", 0.0)))))
        if trace_rows is not None:
            px, py, pth = info.get("pose_m", (env._x_m, env._y_m, env._psi_rad))
            trace_rows.append(
                {
                    "step": int(steps),
                    "x_m": float(px),
                    "y_m": float(py),
                    "theta_rad": float(pth),
                    "v_m_s": float(info.get("v_m_s", env._v_m_s)),
                    "delta_rad": float(info.get("delta_rad", env._delta_rad)),
                    "delta_dot_rad_s": float(delta_dot),
                    "a_m_s2": float(accel),
                    "od_m": float(info.get("od_m", float("nan"))),
                    "collision": bool(info.get("collision", False)),
                    "reached": bool(info.get("reached", False)),
                    "stuck": bool(info.get("stuck", False)),
                }
            )
        if info.get("reached"):
            reached = True
            break

    if time_mode == "rollout":
        inference_time_s = float(time.perf_counter() - t_rollout0)

    if trace_path is not None and trace_rows is not None:
        trace_path = Path(trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(trace_rows).to_csv(trace_path, index=False)

    controls = None
    if t_series is not None and v_series is not None and delta_series is not None:
        controls = ControlTrace(
            t_s=np.asarray(t_series, dtype=np.float64),
            v_m_s=np.asarray(v_series, dtype=np.float64),
            delta_rad=np.asarray(delta_series, dtype=np.float64),
        )
    return RolloutResult(
        path_xy_cells=path,
        compute_time_s=float(inference_time_s),
        reached=bool(reached),
        steps=int(steps),
        path_time_s=float(steps) * dt_s,
        controls=controls,
    )


def infer_checkpoint_obs_dim(path: Path) -> int:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict) or "q_state_dict" not in payload:
        raise ValueError(f"Unsupported checkpoint format: {path}")

    obs_dim = payload.get("obs_dim")
    if isinstance(obs_dim, (int, float)) and int(obs_dim) > 0:
        return int(obs_dim)

    sd = payload["q_state_dict"]
    w = sd.get("net.0.weight")
    if w is None:
        w = sd.get("feature.0.weight")
    if w is None:
        raise ValueError(f"Could not infer observation dim from checkpoint: {path}")
    return int(w.shape[1])


def forest_legacy_obs_transform(obs: np.ndarray) -> np.ndarray:
    """Map current forest observations (11+n_sectors) -> legacy (7+n_sectors)."""
    x = np.asarray(obs, dtype=np.float32).reshape(-1)
    if x.size < 11:
        return x
    return np.concatenate([x[:7], x[11:]]).astype(np.float32, copy=False)


def mean_kpi(kpis: list[KPI]) -> KPI:
    if not kpis:
        nan = float("nan")
        return KPI(
            avg_path_length=nan,
            path_time_s=nan,
            avg_curvature_1_m=nan,
            planning_time_s=nan,
            tracking_time_s=nan,
            num_corners=nan,
            inference_time_s=nan,
            max_corner_deg=nan,
        )
    return KPI(
        avg_path_length=float(np.mean([k.avg_path_length for k in kpis])),
        path_time_s=float(np.mean([k.path_time_s for k in kpis])),
        avg_curvature_1_m=float(np.mean([k.avg_curvature_1_m for k in kpis])),
        planning_time_s=float(np.mean([k.planning_time_s for k in kpis])),
        tracking_time_s=float(np.mean([k.tracking_time_s for k in kpis])),
        num_corners=float(np.mean([k.num_corners for k in kpis])),
        inference_time_s=float(np.mean([k.inference_time_s for k in kpis])),
        max_corner_deg=float(np.mean([k.max_corner_deg for k in kpis])),
    )


def smooth_path(path: list[tuple[float, float]], *, iterations: int) -> list[tuple[float, float]]:
    if not path:
        return []
    pts = np.array(path, dtype=np.float32)
    sm = chaikin_smooth(pts, iterations=max(0, int(iterations)))
    return [(float(x), float(y)) for x, y in sm]


def plot_env(ax: plt.Axes, grid: np.ndarray, *, title: str) -> None:
    ax.imshow(grid, origin="lower", cmap="gray_r", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax.set_ylim(-0.5, grid.shape[0] - 0.5)
    ax.set_aspect("equal")
    h, w = grid.shape
    size = int(max(h, w))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=7, integer=True))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=7, integer=True))
    ax.tick_params(axis="both", labelsize=7)
    ax.tick_params(axis="x", labelrotation=45)

    if size <= 60:
        ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
        ax.grid(True, which="minor", alpha=0.18, linewidth=0.35)
        ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
    else:
        ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
        ax.grid(False, which="minor")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")


def draw_vehicle_boxes(
    ax: plt.Axes,
    trace: PathTrace,
    *,
    length_cells: float,
    width_cells: float,
    color: str,
) -> None:
    if not (float(length_cells) > 0.0 and float(width_cells) > 0.0):
        return
    path = trace.path_xy_cells
    if len(path) < 2:
        return

    stride = max(1, int(len(path) / 18))
    hl = 0.5 * float(length_cells)
    hw = 0.5 * float(width_cells)
    alpha = 0.28 if trace.success else 0.18
    ls = "-" if trace.success else ":"

    prev_theta: float | None = None
    for i in range(0, len(path), stride):
        x, y = path[i]
        if i < len(path) - 1:
            x2, y2 = path[i + 1]
            dx = float(x2) - float(x)
            dy = float(y2) - float(y)
        else:
            x2, y2 = path[i - 1]
            dx = float(x) - float(x2)
            dy = float(y) - float(y2)

        if abs(dx) + abs(dy) < 1e-9:
            theta = float(prev_theta) if prev_theta is not None else 0.0
        else:
            theta = float(math.atan2(dy, dx))
        prev_theta = float(theta)

        c = float(math.cos(theta))
        s = float(math.sin(theta))
        corners = [
            (float(x) + c * hl - s * hw, float(y) + s * hl + c * hw),
            (float(x) + c * hl - s * (-hw), float(y) + s * hl + c * (-hw)),
            (float(x) + c * (-hl) - s * (-hw), float(y) + s * (-hl) + c * (-hw)),
            (float(x) + c * (-hl) - s * hw, float(y) + s * (-hl) + c * hw),
        ]
        poly = mpatches.Polygon(
            corners,
            closed=True,
            fill=False,
            edgecolor=color,
            linewidth=0.6,
            alpha=float(alpha),
            linestyle=ls,
        )
        ax.add_patch(poly)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run inference and generate Fig.12 + Table II-style KPIs.")
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON config file. Supports a combined file with {train:{...}, infer:{...}}. CLI flags override config.",
    )
    ap.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Config profile name under configs/ (e.g. forest_a_3000 -> configs/forest_a_3000.json). Overrides configs/config.json.",
    )
    ap.add_argument(
        "--envs",
        nargs="*",
        default=list(FOREST_ENV_ORDER),
        help="Subset of envs: forest_a forest_b forest_c forest_d",
    )
    ap.add_argument(
        "--models",
        type=Path,
        default=Path("outputs"),
        help="Model source: experiment name/dir, run dir, or models dir.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs"),
        help=(
            "Output experiment name/dir. If this resolves to the same experiment as --models, "
            "results are stored under that training run directory."
        ),
    )
    ap.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="If --out/--models is a bare name, store/read it under this directory.",
    )
    ap.add_argument(
        "--timestamp-runs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write into <experiment>/<timestamp>/ (or <train_run>/infer/<timestamp>/) to avoid mixing outputs.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=int, default=5, help="Averaging runs for stochastic methods.")
    ap.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show an inference progress bar (default: on when running in a TTY).",
    )
    ap.add_argument(
        "--plot-run-idx",
        type=int,
        default=0,
        help=(
            "When --random-start-goal is enabled, plot this sample index in fig12_paths.png so all algorithms share "
            "the same (start,goal) pair (default: 0)."
        ),
    )
    ap.add_argument(
        "--plot-pair-runs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Forest-only: when --rand-two-suites is enabled, write one 2-panel path figure per run index "
            "(short + long) so each image contains a short and a long random pair."
        ),
    )
    ap.add_argument(
        "--plot-pair-runs-max",
        type=int,
        default=10,
        help="Maximum number of per-run short+long figures to write when --plot-pair-runs is enabled (<=0 disables cap).",
    )
    ap.add_argument(
        "--baselines",
        nargs="*",
        default=[],
        help="Optional baselines to evaluate: hybrid_astar rrt_star (or 'all'). Default: none.",
    )
    ap.add_argument(
        "--rl-algos",
        nargs="+",
        default=["mlp-dqn"],
        help=(
            "RL algorithms to evaluate: mlp-dqn mlp-ddqn mlp-pddqn cnn-dqn cnn-ddqn cnn-pddqn (or 'all'). "
            "Legacy aliases: dqn ddqn iddqn cnn-iddqn. Default: mlp-dqn."
        ),
    )
    ap.add_argument(
        "--skip-rl",
        action="store_true",
        help="Skip loading/running RL agents (useful for baseline-only evaluation).",
    )
    ap.add_argument("--baseline-timeout", type=float, default=5.0, help="Planner timeout (seconds).")
    ap.add_argument("--hybrid-max-nodes", type=int, default=200_000, help="Hybrid A* node budget.")
    ap.add_argument("--rrt-max-iter", type=int, default=5_000, help="RRT* iteration budget.")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--sensor-range", type=int, default=6)
    ap.add_argument(
        "--n-sectors",
        type=int,
        default=36,
        help="Forest lidar sectors (kept for backwards compatibility; not used by the global-map observation).",
    )
    ap.add_argument(
        "--obs-map-size",
        type=int,
        default=12,
        help="Downsampled global-map observation size (applies to both grid and forest envs).",
    )
    ap.add_argument("--cell-size", type=float, default=1.0, help="Grid cell size in meters.")
    ap.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cuda",
        help="Torch device selection (default: cuda).",
    )
    ap.add_argument("--cuda-device", type=int, default=0, help="CUDA device index (when using --device=cuda).")
    ap.add_argument(
        "--score-time-weight",
        type=float,
        default=0.5,
        help=(
            "Time weight (m/s) for the composite planning_cost metric: "
            "planning_cost = (avg_path_length + w * inference_time_s) / max(success_rate, eps)."
        ),
    )
    ap.add_argument(
        "--composite-w-path-time",
        type=float,
        default=1.0,
        help="Weight for path_time_s in composite_score (default: 1.0).",
    )
    ap.add_argument(
        "--composite-w-avg-curvature",
        type=float,
        default=1.0,
        help="Weight for avg_curvature_1_m in composite_score (default: 1.0).",
    )
    ap.add_argument(
        "--composite-w-planning-time",
        type=float,
        default=1.0,
        help="Weight for planning_time_s in composite_score (default: 1.0).",
    )
    ap.add_argument(
        "--kpi-time-mode",
        choices=("rollout", "policy"),
        default="policy",
        help=(
            "How to measure inference_time_s for RL rollouts. "
            "'rollout' includes the full rollout wall-clock time (including env.step); "
            "'policy' measures only action-selection compute time (Q forward + admissibility checks)."
        ),
    )
    ap.add_argument(
        "--forest-adm-horizon",
        type=int,
        default=15,
        help="Forest-only: admissible-action horizon steps for safe/progress-gated rollouts.",
    )
    ap.add_argument(
        "--forest-topk",
        type=int,
        default=10,
        help="Forest-only: try the top-k greedy actions before computing a full admissible-action mask.",
    )
    ap.add_argument(
        "--forest-min-progress-m",
        type=float,
        default=1e-4,
        help="Forest-only: minimum cost-to-go progress required by admissible-action gating.",
    )
    ap.add_argument(
        "--forest-min-od-m",
        type=float,
        default=0.0,
        help="Forest-only: minimum clearance (OD) required by admissible-action gating.",
    )
    ap.add_argument(
        "--forest-baseline-rollout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Forest-only: when baselines are enabled, roll out a tracking controller on the planned "
            "baseline path to report executed-trajectory KPIs (default: enabled). "
            "Disable with --no-forest-baseline-rollout."
        ),
    )
    ap.add_argument(
        "--forest-baseline-mpc-candidates",
        type=int,
        default=256,
        help="Forest-only: continuous control samples per MPC step for baseline tracking.",
    )
    ap.add_argument(
        "--forest-baseline-save-traces",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Forest-only: when --forest-baseline-rollout is enabled, save per-run executed baseline trajectories "
            "(x,y,theta,v,delta,controls,OD) under <run_dir>/traces/ as CSV."
        ),
    )
    ap.add_argument(
        "--random-start-goal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forest-only: evaluate on random start/goal pairs (uses --runs samples per environment).",
    )
    ap.add_argument(
        "--rand-min-cost-m",
        type=float,
        default=6.0,
        help="Forest-only: minimum start→goal cost-to-go (meters) when sampling random pairs.",
    )
    ap.add_argument(
        "--rand-max-cost-m",
        type=float,
        default=0.0,
        help="Forest-only: maximum start→goal cost-to-go (meters) when sampling random pairs (<=0 disables).",
    )
    ap.add_argument(
        "--rand-fixed-prob",
        type=float,
        default=0.0,
        help="Forest-only: probability of using the canonical fixed start/goal instead of a random pair.",
    )
    ap.add_argument(
        "--rand-tries",
        type=int,
        default=200,
        help="Forest-only: rejection-sampling tries per sample when sampling random start/goal pairs.",
    )
    ap.add_argument(
        "--rand-reject-unreachable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Forest-only: when --random-start-goal is enabled, resample until Hybrid A* succeeds "
            "(avoids unreachable start/goal pairs in narrow forests and keeps comparisons meaningful)."
        ),
    )
    ap.add_argument(
        "--rand-reject-max-attempts",
        type=int,
        default=5000,
        help="Forest-only: maximum sampling attempts to find reachable random (start,goal) pairs.",
    )
    ap.add_argument(
        "--rand-two-suites",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Forest-only: when --random-start-goal is enabled, evaluate two random-pair suites (short + long) in one run. "
            "This adds '/short' and '/long' rows to KPI tables and plots."
        ),
    )
    ap.add_argument(
        "--rand-short-min-cost-m",
        type=float,
        default=6.0,
        help="Forest-only: minimum start→goal cost-to-go (meters) for the 'short' random-pair suite.",
    )
    ap.add_argument(
        "--rand-short-max-cost-m",
        type=float,
        default=14.0,
        help="Forest-only: maximum start→goal cost-to-go (meters) for the 'short' random-pair suite (<=0 disables).",
    )
    ap.add_argument(
        "--rand-long-min-cost-m",
        type=float,
        default=18.0,
        help="Forest-only: minimum start→goal cost-to-go (meters) for the 'long' random-pair suite.",
    )
    ap.add_argument(
        "--rand-long-max-cost-m",
        type=float,
        default=0.0,
        help="Forest-only: maximum start→goal cost-to-go (meters) for the 'long' random-pair suite (<=0 disables).",
    )
    ap.add_argument(
        "--self-check",
        action="store_true",
        help="Print CUDA/runtime info and exit (use to verify CUDA setup).",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = build_parser()

    pre_args, _ = ap.parse_known_args(argv)
    try:
        config_path = resolve_config_path(config=getattr(pre_args, "config", None), profile=getattr(pre_args, "profile", None))
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc))
    if config_path is not None:
        cfg_raw = load_json(Path(config_path))
        cfg = select_section(cfg_raw, section="infer")
        if "forest_baseline_controller" in cfg:
            print(
                "Warning: config key 'forest_baseline_controller' is deprecated and ignored; "
                "baseline rollouts always use MPC now.",
                file=sys.stderr,
            )
        apply_config_defaults(ap, cfg, strict=True, allow_unknown_prefixes=("_", "forest_baseline_controller"))

    args = ap.parse_args(argv)
    if int(getattr(args, "plot_run_idx", 0)) < 0:
        raise SystemExit("--plot-run-idx must be >= 0")
    forest_envs = set(FOREST_ENV_ORDER)
    if int(args.max_steps) == 300 and args.envs and all(str(e) in forest_envs for e in args.envs):
        args.max_steps = 600
    canonical_all = ("mlp-dqn", "mlp-ddqn", "mlp-pddqn", "cnn-dqn", "cnn-ddqn", "cnn-pddqn")
    raw_algos = [str(a).lower().strip() for a in (args.rl_algos or [])]
    if any(a == "all" for a in raw_algos):
        raw_algos = list(canonical_all)

    rl_algos: list[str] = []
    unknown = []
    for a in raw_algos:
        try:
            canonical, _arch, _base, _legacy = parse_rl_algo(a)
        except ValueError:
            unknown.append(a)
            continue
        if canonical not in rl_algos:
            rl_algos.append(canonical)

    if unknown:
        raise SystemExit(
            f"Unknown --rl-algos value(s): {', '.join(unknown)}. Choose from: "
            f"{' '.join(canonical_all)} (or 'all'). Legacy aliases: dqn ddqn iddqn cnn-iddqn."
        )
    if not rl_algos:
        raise SystemExit(f"No RL algorithms selected (choose from: {' '.join(canonical_all)}).")
    args.rl_algos = rl_algos

    baseline_aliases = {
        "hybrid_astar": "hybrid_astar",
        "hybrid-a-star": "hybrid_astar",
        "hybrid": "hybrid_astar",
        "ha": "hybrid_astar",
        "rrt_star": "rrt_star",
        "rrt*": "rrt_star",
        "rrt": "rrt_star",
        "ss-rrt*": "rrt_star",
        "ss_rrt_star": "rrt_star",
        "ss_rrt*": "rrt_star",
        "all": "all",
    }
    baselines: list[str] = []
    for raw in args.baselines:
        key = str(raw).strip().lower()
        if not key:
            continue
        mapped = baseline_aliases.get(key)
        if mapped is None:
            raise SystemExit(
                f"Unknown baseline {raw!r}. Options: hybrid_astar, rrt_star, all (aliases: hybrid, rrt, rrt*)."
            )
        if mapped == "all":
            baselines = ["hybrid_astar", "rrt_star"]
            break
        if mapped not in baselines:
            baselines.append(mapped)

    if bool(args.skip_rl) and not baselines:
        raise SystemExit("--skip-rl requires at least one baseline via --baselines (e.g., --baselines all).")

    if bool(getattr(args, "rand_two_suites", False)):
        if not bool(getattr(args, "random_start_goal", False)):
            raise SystemExit("--rand-two-suites requires --random-start-goal.")
        if not args.envs or any(str(e) not in forest_envs for e in args.envs):
            raise SystemExit("--rand-two-suites is forest-only (use e.g. --envs forest_a ...).")
        if int(getattr(args, "runs", 0)) <= 0:
            raise SystemExit("--rand-two-suites requires --runs >= 1.")
        short_min = float(getattr(args, "rand_short_min_cost_m", 0.0))
        short_max = float(getattr(args, "rand_short_max_cost_m", 0.0))
        long_min = float(getattr(args, "rand_long_min_cost_m", 0.0))
        long_max = float(getattr(args, "rand_long_max_cost_m", 0.0))
        if short_max > 0.0 and short_min > short_max:
            raise SystemExit("--rand-short-min-cost-m must be <= --rand-short-max-cost-m (or disable max via <=0).")
        if long_max > 0.0 and long_min > long_max:
            raise SystemExit("--rand-long-min-cost-m must be <= --rand-long-max-cost-m (or disable max via <=0).")

    if bool(getattr(args, "rand_two_suites", False)):
        expanded_envs: list[str] = []
        for e in (args.envs or []):
            base = str(e).split("::", 1)[0].strip()
            if not base:
                continue
            expanded_envs.append(f"{base}::short")
            expanded_envs.append(f"{base}::long")
        args.envs = expanded_envs

    if args.self_check:
        info = torch_runtime_info()
        print(f"torch={info.torch_version}")
        print(f"cuda_available={info.cuda_available}")
        print(f"torch_cuda_version={info.torch_cuda_version}")
        print(f"cuda_device_count={info.device_count}")
        if info.device_names:
            print("cuda_devices=" + ", ".join(info.device_names))
        try:
            device = select_device(device=args.device, cuda_device=args.cuda_device)
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"device_ok={device}")
        return 0

    try:
        device = select_device(device=args.device, cuda_device=args.cuda_device)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    progress = bool(sys.stderr.isatty()) if getattr(args, "progress", None) is None else bool(getattr(args, "progress"))
    tqdm = None
    if progress:
        try:
            from tqdm import tqdm as _tqdm  # type: ignore
        except Exception:
            tqdm = None
        else:
            tqdm = _tqdm

    requested_experiment_dir = resolve_experiment_dir(args.out, runs_root=args.runs_root)
    models_dir: Path | None = None
    if not bool(args.skip_rl):
        models_dir = resolve_models_dir(args.models, runs_root=args.runs_root)

        models_run_dir = models_dir.parent
        models_experiment_dir = models_run_dir.parent

        # If the output points to the same experiment (timestamped runs) or the same run dir (no-timestamp runs),
        # keep inference outputs attached to the training run.
        requested_resolved = requested_experiment_dir.resolve(strict=False)
        models_run_resolved = models_run_dir.resolve(strict=False)
        models_experiment_resolved = models_experiment_dir.resolve(strict=False)

        if requested_resolved == models_run_resolved or requested_resolved == models_experiment_resolved:
            # Keep inference outputs attached to the training run to avoid creating a sibling timestamped run.
            experiment_dir = models_run_dir / "infer"
        else:
            experiment_dir = requested_experiment_dir
    else:
        experiment_dir = requested_experiment_dir

    run_paths = create_run_dir(experiment_dir, timestamp_runs=args.timestamp_runs)
    out_dir = run_paths.run_dir

    (out_dir / "configs").mkdir(parents=True, exist_ok=True)
    args_payload: dict[str, object] = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            args_payload[k] = str(v)
        else:
            args_payload[k] = v
    (out_dir / "configs" / "run.json").write_text(
        json.dumps(
            {
                "kind": "infer",
                "argv": list(sys.argv),
                "experiment_dir": str(run_paths.experiment_dir),
                "run_dir": str(run_paths.run_dir),
                "models_dir": (str(models_dir) if models_dir is not None else None),
                "args": args_payload,
                "torch": asdict(torch_runtime_info()),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    rows: list[dict[str, object]] = []
    rows_runs: list[dict[str, object]] = []
    paths_for_plot: dict[tuple[str, int], dict[str, PathTrace]] = {}
    controls_for_plot: dict[tuple[str, int], dict[str, ControlTrace]] = {}
    plot_meta: dict[tuple[str, int], dict[str, float]] = {}

    for env_name in args.envs:
        env_case = str(env_name)
        suite_tag: str | None = None
        env_base = str(env_case)
        if "::" in env_case:
            env_base, suite_tag_raw = env_case.split("::", 1)
            env_base = str(env_base).strip()
            suite_tag = str(suite_tag_raw).strip() or None

        env_label = f"Env. ({env_base})" if suite_tag is None else f"Env. ({env_base})/{suite_tag}"

        rand_min_cost_m = float(getattr(args, "rand_min_cost_m", 0.0))
        rand_max_cost_m = float(getattr(args, "rand_max_cost_m", 0.0))
        if suite_tag == "short":
            rand_min_cost_m = float(getattr(args, "rand_short_min_cost_m", rand_min_cost_m))
            rand_max_cost_m = float(getattr(args, "rand_short_max_cost_m", rand_max_cost_m))
        elif suite_tag == "long":
            rand_min_cost_m = float(getattr(args, "rand_long_min_cost_m", rand_min_cost_m))
            rand_max_cost_m = float(getattr(args, "rand_long_max_cost_m", rand_max_cost_m))

        spec = get_map_spec(env_base)
        if env_base in FOREST_ENV_ORDER:
            env = AMRBicycleEnv(
                spec,
                max_steps=args.max_steps,
                cell_size_m=0.1,
                sensor_range_m=float(args.sensor_range),
                n_sectors=args.n_sectors,
                obs_map_size=int(args.obs_map_size),
            )
            cell_size_m = 0.1
        else:
            env = AMRGridEnv(
                spec,
                sensor_range=args.sensor_range,
                max_steps=args.max_steps,
                reward=RewardWeights(),
                cell_size=args.cell_size,
                safe_distance=0.6,
                obs_map_size=int(args.obs_map_size),
                terminate_on_collision=False,
            )
            cell_size_m = float(args.cell_size)
        grid = spec.obstacle_grid()

        env_paths_by_run: dict[int, dict[str, PathTrace]] = {}
        base_meta: dict[str, float] = {"cell_size_m": float(cell_size_m)}
        if isinstance(env, AMRBicycleEnv):
            base_meta["goal_tol_cells"] = float(env.goal_tolerance_m) / float(cell_size_m)
            fp = forest_two_circle_footprint()
            base_meta["veh_length_cells"] = float(fp.length) / float(cell_size_m)
            base_meta["veh_width_cells"] = float(fp.width) / float(cell_size_m)
        else:
            base_meta["goal_tol_cells"] = 0.5
            base_meta["veh_length_cells"] = 0.0
            base_meta["veh_width_cells"] = 0.0

        plot_run_idx = int(getattr(args, "plot_run_idx", 0))
        plot_run_indices: list[int] = [int(plot_run_idx)]
        multi_pair_plot = (
            bool(getattr(args, "random_start_goal", False))
            and isinstance(env, AMRBicycleEnv)
            and int(args.runs) >= 4
            and int(len(args.envs)) == 1
        )
        if multi_pair_plot:
            plot_run_indices = [(int(plot_run_idx) + k) % int(args.runs) for k in range(4)]

        # Plotting: store path traces for specific run indices to keep memory bounded.
        # - `plot_run_indices` drives the main Fig.12/Fig.13 panels.
        # - `plot_pair_runs` wants per-run path figures, but doesn't require control traces.
        path_run_indices: set[int] = set(plot_run_indices)
        control_run_indices: set[int] = set(plot_run_indices)
        if (
            bool(getattr(args, "plot_pair_runs", False))
            and bool(getattr(args, "random_start_goal", False))
            and bool(getattr(args, "rand_two_suites", False))
            and int(args.runs) > 0
        ):
            per_run_cap = int(getattr(args, "plot_pair_runs_max", 10))
            per_run_n = int(args.runs)
            if per_run_cap > 0:
                per_run_n = min(int(per_run_n), int(per_run_cap))
            path_run_indices.update(range(int(per_run_n)))

        for idx in sorted(path_run_indices):
            env_paths_by_run.setdefault(int(idx), {})

        # Optional: sample a fixed set of (start, goal) pairs for fair random-start/goal evaluation.
        reset_options_list: list[dict[str, object] | None] = [None] * int(max(0, int(args.runs)))
        precomputed_hybrid_paths: list[PlannerResult] | None = None
        plot_start_xy = tuple(spec.start_xy)
        plot_goal_xy = tuple(spec.goal_xy)
        if bool(getattr(args, "random_start_goal", False)) and isinstance(env, AMRBicycleEnv) and int(args.runs) > 0:
            rand_max = None if float(rand_max_cost_m) <= 0.0 else float(rand_max_cost_m)
            if plot_run_idx >= int(args.runs):
                raise SystemExit(
                    f"--plot-run-idx={plot_run_idx} must be < --runs={int(args.runs)} when --random-start-goal is enabled."
                )
            reset_options_list = []
            reject_unreachable = bool(getattr(args, "rand_reject_unreachable", False))
            max_attempts = max(1, int(getattr(args, "rand_reject_max_attempts", 5000)))
            if reject_unreachable:
                precomputed_hybrid_paths = []
                grid_map = grid_map_from_obstacles(grid_y0_bottom=grid, cell_size_m=float(cell_size_m))
                params = default_ackermann_params(
                    wheelbase_m=float(env.model.wheelbase_m),
                    delta_max_rad=float(env.model.delta_max_rad),
                    v_max_m_s=float(env.model.v_max_m_s),
                )
                footprint = forest_two_circle_footprint()
                goal_xy_tol_m = float(env.goal_tolerance_m)
                goal_theta_tol_rad = float(env.goal_angle_tolerance_rad)

            sample_pbar = None
            if tqdm is not None:
                sample_pbar = tqdm(
                    total=int(args.runs),
                    desc=f"Sample pairs {env_label}",
                    unit="pair",
                    dynamic_ncols=True,
                    leave=False,
                )

            attempts = 0
            try:
                while len(reset_options_list) < int(args.runs) and attempts < max_attempts:
                    env.reset(
                        seed=int(args.seed) + 90_000 + int(attempts),
                        options={
                            "random_start_goal": True,
                            "rand_min_cost_m": float(rand_min_cost_m),
                            "rand_max_cost_m": rand_max,
                            "rand_fixed_prob": float(args.rand_fixed_prob),
                            "rand_tries": int(args.rand_tries),
                        },
                    )

                    start_xy = (int(env.start_xy[0]), int(env.start_xy[1]))
                    goal_xy = (int(env.goal_xy[0]), int(env.goal_xy[1]))

                    accept = True
                    # When the sampling constraints are too strict, the env falls back to the canonical
                    # (start,goal) pair after exhausting `rand_tries`. That defeats the purpose of
                    # random-pair evaluation and also breaks the short/long suite separation.
                    if float(getattr(args, "rand_fixed_prob", 0.0)) <= 0.0:
                        if start_xy == (int(spec.start_xy[0]), int(spec.start_xy[1])) and goal_xy == (
                            int(spec.goal_xy[0]),
                            int(spec.goal_xy[1]),
                        ):
                            accept = False
                        if accept:
                            cost0 = float(env._cost_to_goal_m[int(start_xy[1]), int(start_xy[0])])
                            if not math.isfinite(cost0):
                                accept = False
                            elif float(cost0) + 1e-6 < float(rand_min_cost_m):
                                accept = False
                            elif rand_max is not None and float(rand_max) > 0.0 and float(cost0) - 1e-6 > float(rand_max):
                                accept = False

                    if accept and reject_unreachable:
                        res = plan_hybrid_astar(
                            grid_map=grid_map,
                            footprint=footprint,
                            params=params,
                            start_xy=start_xy,
                            goal_xy=goal_xy,
                            goal_theta_rad=0.0,
                            start_theta_rad=None,
                            goal_xy_tol_m=goal_xy_tol_m,
                            goal_theta_tol_rad=goal_theta_tol_rad,
                            timeout_s=float(args.baseline_timeout),
                            max_nodes=int(args.hybrid_max_nodes),
                        )
                        if not bool(res.success):
                            accept = False
                        elif precomputed_hybrid_paths is not None:
                            precomputed_hybrid_paths.append(res)

                    if accept:
                        opts: dict[str, object] = {"start_xy": start_xy, "goal_xy": goal_xy}
                        reset_options_list.append(opts)
                        if sample_pbar is not None:
                            sample_pbar.update(1)

                    attempts += 1
                    if sample_pbar is not None and (attempts % 25 == 0):
                        sample_pbar.set_postfix_str(f"attempts={attempts}")
            finally:
                if sample_pbar is not None:
                    sample_pbar.close()

            if len(reset_options_list) < int(args.runs):
                raise RuntimeError(
                    f"Could not sample {int(args.runs)} reachable random (start,goal) pairs for {env_name!r} "
                    f"after {attempts} attempts (rand_min_cost_m={float(rand_min_cost_m):.2f}, rand_max_cost_m={rand_max}). "
                    "Try increasing --rand-tries, adjusting the cost bounds, or disabling screening via --no-rand-reject-unreachable."
                )
            if reset_options_list:
                plot_start_xy = tuple(reset_options_list[plot_run_idx]["start_xy"])  # type: ignore[arg-type]
                plot_goal_xy = tuple(reset_options_list[plot_run_idx]["goal_xy"])  # type: ignore[arg-type]

        use_random_pairs = bool(getattr(args, "random_start_goal", False)) and bool(reset_options_list) and reset_options_list[0] is not None
        env_pbar = None
        if tqdm is not None:
            total_rollouts = 0
            if not bool(args.skip_rl):
                total_rollouts += int(args.runs) * int(len(args.rl_algos))
            if "hybrid_astar" in baselines:
                total_rollouts += int(args.runs) if use_random_pairs else 1
            if "rrt_star" in baselines:
                total_rollouts += int(args.runs)
            if total_rollouts > 0:
                env_pbar = tqdm(
                    total=int(total_rollouts),
                    desc=f"Infer {env_label}",
                    unit="rollout",
                    dynamic_ncols=True,
                    leave=True,
                )

        meta_run_indices = sorted(path_run_indices)
        if not reset_options_list or reset_options_list[0] is None:
            panel_start_goal: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {
                int(idx): ((int(spec.start_xy[0]), int(spec.start_xy[1])), (int(spec.goal_xy[0]), int(spec.goal_xy[1])))
                for idx in meta_run_indices
            }
        else:
            panel_start_goal = {
                int(idx): (
                    tuple(reset_options_list[int(idx)]["start_xy"]),  # type: ignore[arg-type]
                    tuple(reset_options_list[int(idx)]["goal_xy"]),  # type: ignore[arg-type]
                )
                for idx in meta_run_indices
            }

        for idx, (sp_xy, gp_xy) in panel_start_goal.items():
            meta = dict(base_meta)
            meta["plot_start_x"] = float(sp_xy[0])
            meta["plot_start_y"] = float(sp_xy[1])
            meta["plot_goal_x"] = float(gp_xy[0])
            meta["plot_goal_y"] = float(gp_xy[1])
            meta["plot_run_idx"] = float(idx)
            plot_meta[(env_name, int(idx))] = meta

        if not bool(args.skip_rl):
            # Load trained models
            env_obs_dim = int(env.observation_space.shape[0])
            n_actions = int(env.action_space.n)
            agent_cfg = AgentConfig()

            algo_label = {
                "mlp-dqn": "MLP-DQN",
                "mlp-ddqn": "MLP-DDQN",
                "mlp-pddqn": "MLP-PDDQN",
                "cnn-dqn": "CNN-DQN",
                "cnn-ddqn": "CNN-DDQN",
                "cnn-pddqn": "CNN-PDDQN",
            }
            algo_seed_offset = {
                "mlp-dqn": 20_000,
                "mlp-ddqn": 30_000,
                "mlp-pddqn": 60_000,
                "cnn-dqn": 40_000,
                "cnn-ddqn": 50_000,
                "cnn-pddqn": 70_000,
            }

            def resolve_model_path(algo: str) -> Path:
                p = models_dir / env_base / f"{algo}.pt"
                if p.exists():
                    return p
                legacy = {
                    "mlp-dqn": "dqn",
                    "mlp-ddqn": "ddqn",
                    # Back-compat: older runs saved Polyak-DDQN as iddqn/cnn-iddqn.
                    "mlp-pddqn": "iddqn",
                    "cnn-pddqn": "cnn-iddqn",
                }.get(str(algo))
                if legacy is not None:
                    p_legacy = models_dir / env_base / f"{legacy}.pt"
                    if p_legacy.exists():
                        return p_legacy
                return p

            algo_paths = {str(a): resolve_model_path(str(a)) for a in args.rl_algos}
            missing = [str(p) for p in algo_paths.values() if not p.exists()]
            if missing:
                exp = ", ".join(str(p) for p in algo_paths.values())
                raise FileNotFoundError(
                    f"Missing model(s) for env {env_base!r}. Expected: {exp}. "
                    "Point --models at a training run (or an experiment name/dir with a latest run)."
                )

            # Each arch (MLP / CNN) may have a different effective obs_dim
            # (MLP strips the EDT channel). The agent constructor and load()
            # handle this automatically, so we just pass env_obs_dim.
            obs_dim = env_obs_dim
            obs_transform = None

            agents: dict[str, DQNFamilyAgent] = {}
            for algo, path in algo_paths.items():
                a = DQNFamilyAgent(str(algo), obs_dim, n_actions, config=agent_cfg, seed=args.seed, device=device)
                a.load(path)
                agents[str(algo)] = a

            for algo in args.rl_algos:
                algo_key = str(algo)
                pretty = algo_label.get(algo_key, algo_key.upper())
                seed_base = int(algo_seed_offset.get(algo_key, 30_000))

                algo_kpis: list[KPI] = []
                algo_times: list[float] = []
                algo_success = 0
                for i in range(int(args.runs)):
                    roll = rollout_agent(
                        env,
                        agents[algo_key],
                        max_steps=args.max_steps,
                        seed=int(args.seed) + seed_base + int(i),
                        reset_options=reset_options_list[i] if i < len(reset_options_list) else None,
                        time_mode=str(getattr(args, "kpi_time_mode", "rollout")),
                        obs_transform=obs_transform,
                        forest_adm_horizon=int(args.forest_adm_horizon),
                        forest_topk=int(args.forest_topk),
                        forest_min_od_m=float(args.forest_min_od_m),
                        forest_min_progress_m=float(args.forest_min_progress_m),
                        collect_controls=bool(int(i) in control_run_indices),
                    )
                    algo_times.append(float(roll.compute_time_s))
                    if int(i) in path_run_indices:
                        env_paths_by_run[int(i)][pretty] = PathTrace(path_xy_cells=roll.path_xy_cells, success=bool(roll.reached))
                    if roll.controls is not None and int(i) in control_run_indices:
                        controls_for_plot.setdefault((env_name, int(i)), {})[str(pretty)] = roll.controls

                    start_xy = (int(spec.start_xy[0]), int(spec.start_xy[1]))
                    goal_xy = (int(spec.goal_xy[0]), int(spec.goal_xy[1]))
                    opts = reset_options_list[i] if i < len(reset_options_list) else None
                    if isinstance(opts, dict) and "start_xy" in opts and "goal_xy" in opts:
                        sx, sy = opts["start_xy"]  # type: ignore[misc]
                        gx, gy = opts["goal_xy"]  # type: ignore[misc]
                        start_xy = (int(sx), int(sy))
                        goal_xy = (int(gx), int(gy))

                    raw_corners = float(num_path_corners(roll.path_xy_cells, angle_threshold_deg=13.0))
                    smoothed = smooth_path(roll.path_xy_cells, iterations=2)
                    smoothed_m = [(float(x) * float(cell_size_m), float(y) * float(cell_size_m)) for x, y in smoothed]
                    run_kpi = KPI(
                        avg_path_length=float(path_length(smoothed)) * float(cell_size_m),
                        path_time_s=float(roll.path_time_s),
                        avg_curvature_1_m=float(avg_abs_curvature(smoothed_m)),
                        planning_time_s=float(roll.compute_time_s),
                        tracking_time_s=0.0,
                        inference_time_s=float(roll.compute_time_s),
                        num_corners=raw_corners,
                        max_corner_deg=float(max_corner_degree(smoothed)),
                    )
                    rows_runs.append(
                        {
                            "Environment": str(env_label),
                            "Algorithm": str(pretty),
                            "run_idx": int(i),
                            "start_x": int(start_xy[0]),
                            "start_y": int(start_xy[1]),
                            "goal_x": int(goal_xy[0]),
                            "goal_y": int(goal_xy[1]),
                            "success_rate": 1.0 if bool(roll.reached) else 0.0,
                            **dict(run_kpi.__dict__),
                        }
                    )
                    if bool(roll.reached):
                        algo_success += 1
                        algo_kpis.append(run_kpi)
                    if env_pbar is not None:
                        env_pbar.set_postfix_str(f"{pretty} run {int(i) + 1}/{int(args.runs)}")
                        env_pbar.update(1)

                k = mean_kpi(algo_kpis)
                k_dict = dict(k.__dict__)
                if algo_times:
                    mean_plan = float(np.mean(algo_times))
                    k_dict["planning_time_s"] = mean_plan
                    k_dict["tracking_time_s"] = 0.0
                    k_dict["inference_time_s"] = mean_plan
                rows.append(
                    {
                        "Environment": str(env_label),
                        "Algorithm": str(pretty),
                        "success_rate": float(algo_success) / float(max(1, int(args.runs))),
                        **k_dict,
                    }
                )

        if baselines:
            grid_map = grid_map_from_obstacles(grid_y0_bottom=grid, cell_size_m=float(cell_size_m))
            params = default_ackermann_params()
            if env_base in FOREST_ENV_ORDER and isinstance(env, AMRBicycleEnv):
                footprint = forest_two_circle_footprint()
                goal_xy_tol_m = float(env.goal_tolerance_m)
                goal_theta_tol_rad = float(env.goal_angle_tolerance_rad)
                start_theta_rad = None
            else:
                footprint = point_footprint(cell_size_m=float(cell_size_m))
                goal_xy_tol_m = float(cell_size_m) * 0.5
                goal_theta_tol_rad = float(math.pi)
                start_theta_rad = 0.0

            def pair_for_run(i: int) -> tuple[tuple[int, int], tuple[int, int], dict[str, object] | None]:
                if use_random_pairs and i < len(reset_options_list) and reset_options_list[i] is not None:
                    opts = reset_options_list[i] or {}
                    sx, sy = opts["start_xy"]  # type: ignore[misc]
                    gx, gy = opts["goal_xy"]  # type: ignore[misc]
                    return (int(sx), int(sy)), (int(gx), int(gy)), opts
                return (int(spec.start_xy[0]), int(spec.start_xy[1])), (int(spec.goal_xy[0]), int(spec.goal_xy[1])), None

            if "hybrid_astar" in baselines:
                ha_kpis: list[KPI] = []
                ha_plan_times: list[float] = []
                ha_track_times: list[float] = []
                ha_total_times: list[float] = []
                ha_success = 0

                n_runs = int(args.runs) if use_random_pairs else 1
                for i in range(n_runs):
                    start_xy, goal_xy, r_opts = pair_for_run(int(i))
                    if precomputed_hybrid_paths is not None and use_random_pairs and int(i) < len(precomputed_hybrid_paths):
                        res = precomputed_hybrid_paths[int(i)]
                    else:
                        res = plan_hybrid_astar(
                            grid_map=grid_map,
                            footprint=footprint,
                            params=params,
                            start_xy=start_xy,
                            goal_xy=goal_xy,
                            goal_theta_rad=0.0,
                            start_theta_rad=start_theta_rad,
                            goal_xy_tol_m=goal_xy_tol_m,
                            goal_theta_tol_rad=goal_theta_tol_rad,
                            timeout_s=float(args.baseline_timeout),
                            max_nodes=int(args.hybrid_max_nodes),
                        )
                    ha_exec_path = list(res.path_xy_cells)
                    ha_reached = bool(res.success)
                    ha_track_time_s = 0.0
                    ha_path_time_s = float("nan")
                    if bool(res.success) and isinstance(env, AMRBicycleEnv) and bool(getattr(args, "forest_baseline_rollout", False)):
                        trace_path = None
                        if bool(getattr(args, "forest_baseline_save_traces", False)):
                            trace_path = out_dir / "traces" / f"{_safe_slug(env_case)}__Hybrid_A__run{int(i)}.csv"
                        roll = rollout_tracked_path_mpc(
                            env,
                            ha_exec_path,
                            max_steps=args.max_steps,
                            seed=args.seed + 30_000 + i,
                            reset_options=r_opts,
                            time_mode=str(getattr(args, "kpi_time_mode", "policy")),
                            trace_path=trace_path,
                            n_candidates=int(getattr(args, "forest_baseline_mpc_candidates", 256)),
                            collect_controls=bool(int(i) in control_run_indices),
                        )
                        ha_exec_path = list(roll.path_xy_cells)
                        ha_track_time_s = float(roll.compute_time_s)
                        ha_reached = bool(roll.reached)
                        ha_path_time_s = float(roll.path_time_s)
                        if int(i) in control_run_indices and roll.controls is not None:
                            controls_for_plot.setdefault((env_name, int(i)), {})["Hybrid A*"] = roll.controls

                    ha_plan_times.append(float(res.time_s))
                    ha_track_times.append(float(ha_track_time_s))
                    ha_total_times.append(float(res.time_s) + float(ha_track_time_s))
                    if int(i) in path_run_indices:
                        env_paths_by_run[int(i)]["Hybrid A*"] = PathTrace(path_xy_cells=ha_exec_path, success=bool(ha_reached))

                    raw_corners = float(num_path_corners(ha_exec_path, angle_threshold_deg=13.0))
                    smoothed = smooth_path(ha_exec_path, iterations=2)
                    smoothed_m = [(float(x) * float(cell_size_m), float(y) * float(cell_size_m)) for x, y in smoothed]
                    if not math.isfinite(float(ha_path_time_s)) and isinstance(env, AMRBicycleEnv):
                        ha_path_time_s = float(path_length(smoothed_m)) / max(1e-9, float(env.model.v_max_m_s))
                    run_kpi = KPI(
                        avg_path_length=float(path_length(smoothed)) * float(cell_size_m),
                        path_time_s=float(ha_path_time_s),
                        avg_curvature_1_m=float(avg_abs_curvature(smoothed_m)),
                        planning_time_s=float(res.time_s),
                        tracking_time_s=float(ha_track_time_s),
                        inference_time_s=float(res.time_s) + float(ha_track_time_s),
                        num_corners=raw_corners,
                        max_corner_deg=float(max_corner_degree(smoothed)),
                    )
                    rows_runs.append(
                        {
                            "Environment": str(env_label),
                            "Algorithm": "Hybrid A*",
                            "run_idx": int(i),
                            "start_x": int(start_xy[0]),
                            "start_y": int(start_xy[1]),
                            "goal_x": int(goal_xy[0]),
                            "goal_y": int(goal_xy[1]),
                            "success_rate": 1.0 if bool(ha_reached) else 0.0,
                            **dict(run_kpi.__dict__),
                        }
                    )
                    if bool(ha_reached) and ha_exec_path:
                        ha_success += 1
                        ha_kpis.append(run_kpi)
                    if env_pbar is not None:
                        env_pbar.set_postfix_str(f"Hybrid A* run {int(i) + 1}/{int(n_runs)}")
                        env_pbar.update(1)

                k = mean_kpi(ha_kpis)
                k_dict = dict(k.__dict__)
                if ha_plan_times:
                    k_dict["planning_time_s"] = float(np.mean(ha_plan_times))
                if ha_track_times:
                    k_dict["tracking_time_s"] = float(np.mean(ha_track_times))
                if ha_total_times:
                    k_dict["inference_time_s"] = float(np.mean(ha_total_times))
                rows.append(
                    {
                        "Environment": str(env_label),
                        "Algorithm": "Hybrid A*",
                        "success_rate": float(ha_success) / float(max(1, int(n_runs))),
                        **k_dict,
                    }
                )

            if "rrt_star" in baselines:
                rrt_kpis: list[KPI] = []
                rrt_plan_times: list[float] = []
                rrt_track_times: list[float] = []
                rrt_total_times: list[float] = []
                rrt_success = 0
                for i in range(args.runs):
                    start_xy, goal_xy, r_opts = pair_for_run(int(i))
                    res = plan_rrt_star(
                        grid_map=grid_map,
                        footprint=footprint,
                        params=params,
                        start_xy=start_xy,
                        goal_xy=goal_xy,
                        goal_theta_rad=0.0,
                        start_theta_rad=start_theta_rad,
                        goal_xy_tol_m=goal_xy_tol_m,
                        goal_theta_tol_rad=goal_theta_tol_rad,
                        timeout_s=float(args.baseline_timeout),
                        max_iter=int(args.rrt_max_iter),
                        seed=args.seed + 30_000 + i,
                    )
                    exec_path = list(res.path_xy_cells)
                    reached = bool(res.success)
                    track_time_s = 0.0
                    path_time_s = float("nan")
                    if bool(res.success) and isinstance(env, AMRBicycleEnv) and bool(getattr(args, "forest_baseline_rollout", False)):
                        trace_path = None
                        if bool(getattr(args, "forest_baseline_save_traces", False)):
                            trace_path = out_dir / "traces" / f"{_safe_slug(env_case)}__RRT__run{int(i)}.csv"
                        roll = rollout_tracked_path_mpc(
                            env,
                            exec_path,
                            max_steps=args.max_steps,
                            seed=args.seed + 40_000 + i,
                            reset_options=r_opts,
                            time_mode=str(getattr(args, "kpi_time_mode", "policy")),
                            trace_path=trace_path,
                            n_candidates=int(getattr(args, "forest_baseline_mpc_candidates", 256)),
                            collect_controls=bool(int(i) in control_run_indices),
                        )
                        exec_path = list(roll.path_xy_cells)
                        track_time_s = float(roll.compute_time_s)
                        reached = bool(roll.reached)
                        path_time_s = float(roll.path_time_s)
                        if int(i) in control_run_indices and roll.controls is not None:
                            controls_for_plot.setdefault((env_name, int(i)), {})["RRT*"] = roll.controls

                    rrt_plan_times.append(float(res.time_s))
                    rrt_track_times.append(float(track_time_s))
                    rrt_total_times.append(float(res.time_s) + float(track_time_s))
                    if int(i) in path_run_indices:
                        env_paths_by_run[int(i)]["RRT*"] = PathTrace(path_xy_cells=exec_path, success=bool(reached))

                    raw_corners = float(num_path_corners(exec_path, angle_threshold_deg=13.0))
                    smoothed = smooth_path(exec_path, iterations=2)
                    smoothed_m = [(float(x) * float(cell_size_m), float(y) * float(cell_size_m)) for x, y in smoothed]
                    if not math.isfinite(float(path_time_s)) and isinstance(env, AMRBicycleEnv):
                        path_time_s = float(path_length(smoothed_m)) / max(1e-9, float(env.model.v_max_m_s))
                    run_kpi = KPI(
                        avg_path_length=float(path_length(smoothed)) * float(cell_size_m),
                        path_time_s=float(path_time_s),
                        avg_curvature_1_m=float(avg_abs_curvature(smoothed_m)),
                        planning_time_s=float(res.time_s),
                        tracking_time_s=float(track_time_s),
                        inference_time_s=float(res.time_s) + float(track_time_s),
                        num_corners=raw_corners,
                        max_corner_deg=float(max_corner_degree(smoothed)),
                    )
                    rows_runs.append(
                        {
                            "Environment": str(env_label),
                            "Algorithm": "RRT*",
                            "run_idx": int(i),
                            "start_x": int(start_xy[0]),
                            "start_y": int(start_xy[1]),
                            "goal_x": int(goal_xy[0]),
                            "goal_y": int(goal_xy[1]),
                            "success_rate": 1.0 if bool(reached) else 0.0,
                            **dict(run_kpi.__dict__),
                        }
                    )
                    if bool(reached) and exec_path:
                        rrt_success += 1
                        rrt_kpis.append(run_kpi)
                    if env_pbar is not None:
                        env_pbar.set_postfix_str(f"RRT* run {int(i) + 1}/{int(args.runs)}")
                        env_pbar.update(1)

                k = mean_kpi(rrt_kpis)
                k_dict = dict(k.__dict__)
                if rrt_plan_times:
                    k_dict["planning_time_s"] = float(np.mean(rrt_plan_times))
                if rrt_track_times:
                    k_dict["tracking_time_s"] = float(np.mean(rrt_track_times))
                if rrt_total_times:
                    k_dict["inference_time_s"] = float(np.mean(rrt_total_times))
                rows.append(
                    {
                        "Environment": str(env_label),
                        "Algorithm": "RRT*",
                        "success_rate": float(rrt_success) / float(max(1, int(args.runs))),
                        **k_dict,
                    }
                )

        if env_pbar is not None:
            env_pbar.close()
        for run_idx, run_paths in env_paths_by_run.items():
            paths_for_plot[(env_name, int(run_idx))] = dict(run_paths)

    table = pd.DataFrame(rows_runs)
    # Pretty column order
    table = table[
        [
            "Environment",
            "Algorithm",
            "run_idx",
            "start_x",
            "start_y",
            "goal_x",
            "goal_y",
            "success_rate",
            "avg_path_length",
            "path_time_s",
            "avg_curvature_1_m",
            "planning_time_s",
            "tracking_time_s",
            "num_corners",
            "inference_time_s",
            "max_corner_deg",
        ]
    ]
    table = table.copy()

    # Composite metric (lower is better): combines path length and compute time,
    # then penalizes non-reaching behavior via success_rate.
    w_t = float(args.score_time_weight)
    sr_raw = pd.to_numeric(table["success_rate"], errors="coerce").astype(float)
    denom = sr_raw.clip(lower=1e-6)
    base = pd.to_numeric(table["avg_path_length"], errors="coerce").astype(float) + w_t * pd.to_numeric(
        table["inference_time_s"], errors="coerce"
    ).astype(float)
    planning_cost = (base / denom).astype(float)
    planning_cost = planning_cost.where((sr_raw > 0.0) & np.isfinite(base.to_numpy()), other=float("inf"))
    table["planning_cost"] = planning_cost

    # Composite score (lower is better): combines path time, curvature, and planning compute time,
    # then penalizes non-reaching behavior via success_rate.
    w_pt = float(getattr(args, "composite_w_path_time", 1.0))
    w_k = float(getattr(args, "composite_w_avg_curvature", 1.0))
    w_pl = float(getattr(args, "composite_w_planning_time", 1.0))
    w_sum = max(1e-12, float(w_pt + w_k + w_pl))

    def _minmax_norm(s: pd.Series) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce").astype(float)
        v = x.to_numpy(dtype=float, copy=False)
        finite = np.isfinite(v)
        if not bool(finite.any()):
            return pd.Series(np.zeros_like(v, dtype=float), index=x.index)
        mn = float(np.min(v[finite]))
        mx = float(np.max(v[finite]))
        d = float(mx - mn)
        if not math.isfinite(d) or d < 1e-12:
            return pd.Series(np.zeros_like(v, dtype=float), index=x.index)
        out = (v - mn) / d
        out = np.where(finite, out, np.nan)
        return pd.Series(out.astype(float, copy=False), index=x.index)

    group_keys = ["Environment", "run_idx"]
    n_pt = table.groupby(group_keys, sort=False)["path_time_s"].transform(_minmax_norm).fillna(0.0)
    n_k = table.groupby(group_keys, sort=False)["avg_curvature_1_m"].transform(_minmax_norm).fillna(0.0)
    n_pl = table.groupby(group_keys, sort=False)["planning_time_s"].transform(_minmax_norm).fillna(0.0)
    base_score = (w_pt * n_pt + w_k * n_k + w_pl * n_pl) / w_sum
    sr_denom2 = sr_raw.clip(lower=1e-6)
    composite_score = (base_score / sr_denom2).astype(float)
    composite_score = composite_score.where(sr_raw > 0.0, other=float("inf"))
    table["composite_score"] = composite_score

    table["success_rate"] = pd.to_numeric(table["success_rate"], errors="coerce").astype(float).round(3)
    table["avg_path_length"] = table["avg_path_length"].astype(float).round(4)
    table["path_time_s"] = pd.to_numeric(table["path_time_s"], errors="coerce").astype(float).round(4)
    table["avg_curvature_1_m"] = pd.to_numeric(table["avg_curvature_1_m"], errors="coerce").astype(float).round(6)
    table["planning_time_s"] = pd.to_numeric(table["planning_time_s"], errors="coerce").astype(float).round(5)
    table["tracking_time_s"] = pd.to_numeric(table["tracking_time_s"], errors="coerce").astype(float).round(5)
    table["num_corners"] = pd.to_numeric(table["num_corners"], errors="coerce").round(0).astype("Int64")
    table["inference_time_s"] = table["inference_time_s"].astype(float).round(5)
    table["max_corner_deg"] = pd.to_numeric(table["max_corner_deg"], errors="coerce").round(0).astype("Int64")
    table["planning_cost"] = pd.to_numeric(table["planning_cost"], errors="coerce").astype(float).round(3)
    table["composite_score"] = pd.to_numeric(table["composite_score"], errors="coerce").astype(float).round(3)
    table.to_csv(out_dir / "table2_kpis_raw.csv", index=False)

    table_pretty = table.rename(
        columns={
            "Algorithm": "Algorithm name",
            "run_idx": "Run index",
            "start_x": "Start x",
            "start_y": "Start y",
            "goal_x": "Goal x",
            "goal_y": "Goal y",
            "success_rate": "Success rate",
            "avg_path_length": "Average path length (m)",
            "path_time_s": "Path time (s)",
            "avg_curvature_1_m": "Average curvature (1/m)",
            "planning_time_s": "Planning time (s)",
            "tracking_time_s": "Tracking time (s)",
            "num_corners": "Number of path corners",
            "inference_time_s": "Compute time (s)",
            "max_corner_deg": "Max corner degree (°)",
            "planning_cost": "Planning cost (m)",
            "composite_score": "Composite score",
        }
    )
    table_pretty.to_csv(out_dir / "table2_kpis.csv", index=False)
    table_pretty.to_markdown(out_dir / "table2_kpis.md", index=False)

    # Also write the mean KPI table (previous default behavior).
    table_mean = pd.DataFrame(rows)
    table_mean = table_mean[
        [
            "Environment",
            "Algorithm",
            "success_rate",
            "avg_path_length",
            "path_time_s",
            "avg_curvature_1_m",
            "planning_time_s",
            "tracking_time_s",
            "num_corners",
            "inference_time_s",
            "max_corner_deg",
        ]
    ]
    table_mean = table_mean.copy()

    w_t = float(args.score_time_weight)
    sr_raw = pd.to_numeric(table_mean["success_rate"], errors="coerce").astype(float)
    denom = sr_raw.clip(lower=1e-6)
    base = pd.to_numeric(table_mean["avg_path_length"], errors="coerce").astype(float) + w_t * pd.to_numeric(
        table_mean["inference_time_s"], errors="coerce"
    ).astype(float)
    planning_cost = (base / denom).astype(float)
    planning_cost = planning_cost.where((sr_raw > 0.0) & np.isfinite(base.to_numpy()), other=float("inf"))
    table_mean["planning_cost"] = planning_cost

    group_keys = ["Environment"]
    n_pt = table_mean.groupby(group_keys, sort=False)["path_time_s"].transform(_minmax_norm).fillna(0.0)
    n_k = table_mean.groupby(group_keys, sort=False)["avg_curvature_1_m"].transform(_minmax_norm).fillna(0.0)
    n_pl = table_mean.groupby(group_keys, sort=False)["planning_time_s"].transform(_minmax_norm).fillna(0.0)
    base_score = (w_pt * n_pt + w_k * n_k + w_pl * n_pl) / w_sum
    sr_denom2 = sr_raw.clip(lower=1e-6)
    composite_score = (base_score / sr_denom2).astype(float)
    composite_score = composite_score.where(sr_raw > 0.0, other=float("inf"))
    table_mean["composite_score"] = composite_score

    table_mean["success_rate"] = pd.to_numeric(table_mean["success_rate"], errors="coerce").astype(float).round(3)
    table_mean["avg_path_length"] = table_mean["avg_path_length"].astype(float).round(4)
    table_mean["path_time_s"] = pd.to_numeric(table_mean["path_time_s"], errors="coerce").astype(float).round(4)
    table_mean["avg_curvature_1_m"] = pd.to_numeric(table_mean["avg_curvature_1_m"], errors="coerce").astype(float).round(6)
    table_mean["planning_time_s"] = pd.to_numeric(table_mean["planning_time_s"], errors="coerce").astype(float).round(5)
    table_mean["tracking_time_s"] = pd.to_numeric(table_mean["tracking_time_s"], errors="coerce").astype(float).round(5)
    table_mean["num_corners"] = pd.to_numeric(table_mean["num_corners"], errors="coerce").round(0).astype("Int64")
    table_mean["inference_time_s"] = table_mean["inference_time_s"].astype(float).round(5)
    table_mean["max_corner_deg"] = pd.to_numeric(table_mean["max_corner_deg"], errors="coerce").round(0).astype("Int64")
    table_mean["planning_cost"] = pd.to_numeric(table_mean["planning_cost"], errors="coerce").astype(float).round(3)
    table_mean["composite_score"] = pd.to_numeric(table_mean["composite_score"], errors="coerce").astype(float).round(3)
    table_mean.to_csv(out_dir / "table2_kpis_mean_raw.csv", index=False)

    table_mean_pretty = table_mean.rename(
        columns={
            "Algorithm": "Algorithm name",
            "success_rate": "Success rate",
            "avg_path_length": "Average path length (m)",
            "path_time_s": "Path time (s)",
            "avg_curvature_1_m": "Average curvature (1/m)",
            "planning_time_s": "Planning time (s)",
            "tracking_time_s": "Tracking time (s)",
            "num_corners": "Number of path corners",
            "inference_time_s": "Compute time (s)",
            "max_corner_deg": "Max corner degree (掳)",
            "planning_cost": "Planning cost (m)",
            "composite_score": "Composite score",
        }
    )
    table_mean_pretty.to_csv(out_dir / "table2_kpis_mean.csv", index=False)
    table_mean_pretty.to_markdown(out_dir / "table2_kpis_mean.md", index=False)

    # Plot Fig. 12-style paths
    styles = {
        "MLP-DQN": dict(color="tab:blue", linestyle="-", linewidth=2.0),
        "MLP-DDQN": dict(color="tab:orange", linestyle="-", linewidth=2.0),
        "CNN-DQN": dict(color="tab:green", linestyle="-", linewidth=2.0),
        "CNN-DDQN": dict(color="tab:red", linestyle="-", linewidth=2.0),
        "MLP-PDDQN": dict(color="tab:cyan", linestyle="-", linewidth=2.0),
        "CNN-PDDQN": dict(color="tab:pink", linestyle="-", linewidth=2.0),
        # Legacy short labels (treated as MLP variants).
        "DQN": dict(color="tab:blue", linestyle="-", linewidth=2.0),
        "DDQN": dict(color="tab:orange", linestyle="-", linewidth=2.0),
        # Baselines.
        "Hybrid A*": dict(color="tab:purple", linestyle="-", linewidth=2.0),
        "RRT*": dict(color="tab:brown", linestyle="-", linewidth=2.0),
    }

    def write_paths_figure(
        *,
        panels: list[tuple[str, int]],
        out_path: Path,
        suptitle: str,
        multi_pair_titles: bool = False,
    ) -> None:
        n_panels = int(len(panels))
        if n_panels <= 0:
            return
        cols = 1 if n_panels <= 1 else 2
        rows_n = int(math.ceil(float(n_panels) / float(cols)))
        fig, axes = plt.subplots(rows_n, cols, figsize=(5.2 * cols, 5.2 * rows_n))
        axes = np.atleast_1d(axes).ravel()

        for i, (env_name, run_idx) in enumerate(panels):
            ax = axes[i]
            env_base = str(env_name).split("::", 1)[0]
            suite = str(env_name).split("::", 1)[1] if "::" in str(env_name) else ""
            spec = get_map_spec(env_base)
            grid = spec.obstacle_grid()
            title = f"Env. ({env_base})"
            if suite:
                title = f"Env. ({env_base})/{suite}"
            if multi_pair_titles:
                title = f"Env. ({env_base}) #{int(run_idx)}"
            plot_env(ax, grid, title=title)

            meta = plot_meta.get((env_name, int(run_idx))) or plot_meta.get((env_name, 0), {})

            spx = float(meta.get("plot_start_x", float(spec.start_xy[0])))
            spy = float(meta.get("plot_start_y", float(spec.start_xy[1])))
            gpx = float(meta.get("plot_goal_x", float(spec.goal_xy[0])))
            gpy = float(meta.get("plot_goal_y", float(spec.goal_xy[1])))

            ax.scatter(
                [spx],
                [spy],
                marker="*",
                s=140,
                color="blue",
                label="Start",
            )
            ax.text(spx - 1.0, spy - 1.0, "SP", fontsize=9, color="black")
            ax.scatter(
                [gpx],
                [gpy],
                marker="*",
                s=140,
                color="red",
                label="Goal",
            )
            ax.text(gpx - 1.0, gpy - 1.0, "TP", fontsize=9, color="black")

            tol = float(meta.get("goal_tol_cells", 0.0))
            if tol > 0.0:
                ax.add_patch(
                    mpatches.Circle(
                        (float(gpx), float(gpy)),
                        radius=float(tol),
                        fill=False,
                        edgecolor="crimson",
                        linestyle="--",
                        linewidth=1.8,
                        alpha=0.95,
                        zorder=6,
                    )
                )

            env_paths = paths_for_plot.get((env_name, int(run_idx)), {})
            for algo_name, trace in env_paths.items():
                path = trace.path_xy_cells
                if not path:
                    continue
                pts = np.array(path, dtype=np.float32)
                pts_s = chaikin_smooth(pts, iterations=2)
                style = styles.get(algo_name, dict(color="black", linestyle="-", linewidth=1.5))
                label = algo_name if trace.success else f"{algo_name} (fail)"
                alpha = 1.0 if trace.success else 0.55
                ax.plot(pts_s[:, 0], pts_s[:, 1], label=label, alpha=alpha, **style)
                end_marker = "o" if trace.success else "x"
                ax.scatter(
                    [float(pts_s[-1, 0])],
                    [float(pts_s[-1, 1])],
                    marker=end_marker,
                    s=28,
                    color=style["color"],
                    label="_nolegend_",
                )

                if float(meta.get("veh_length_cells", 0.0)) > 0.0 and float(meta.get("veh_width_cells", 0.0)) > 0.0:
                    draw_vehicle_boxes(
                        ax,
                        trace,
                        length_cells=float(meta["veh_length_cells"]),
                        width_cells=float(meta["veh_width_cells"]),
                        color=str(style["color"]),
                    )

            ax.legend(fontsize=8, loc="lower right")

        for ax in axes[n_panels:]:
            ax.axis("off")

        fig.suptitle(str(suptitle))
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    def write_controls_figure(
        *,
        panels: list[tuple[str, int]],
        out_path: Path,
        suptitle: str,
        multi_pair_titles: bool = False,
    ) -> None:
        n_panels = int(len(panels))
        if n_panels <= 0:
            return
        if not any(bool(controls_for_plot.get((env_name, int(run_idx)), {})) for env_name, run_idx in panels):
            return

        fig, axes = plt.subplots(n_panels, 2, figsize=(10.8, 3.2 * n_panels), squeeze=False)

        for i, (env_name, run_idx) in enumerate(panels):
            ax_v = axes[i, 0]
            ax_d = axes[i, 1]

            env_base = str(env_name).split("::", 1)[0]
            suite = str(env_name).split("::", 1)[1] if "::" in str(env_name) else ""
            title = f"Env. ({env_base})"
            if suite:
                title = f"Env. ({env_base})/{suite}"
            if multi_pair_titles:
                title = f"Env. ({env_base}) #{int(run_idx)}"

            ctrl = controls_for_plot.get((env_name, int(run_idx)), {})
            env_paths = paths_for_plot.get((env_name, int(run_idx)), {})
            if not ctrl:
                ax_v.axis("off")
                ax_d.axis("off")
                ax_v.text(0.5, 0.5, f"{title}\n(no control traces)", ha="center", va="center", fontsize=9)
                continue

            for algo_name, tr in ctrl.items():
                style = styles.get(algo_name, dict(color="black", linestyle="-", linewidth=1.5))
                ok = True
                if algo_name in env_paths:
                    ok = bool(env_paths[algo_name].success)
                label = algo_name if ok else f"{algo_name} (fail)"
                alpha = 1.0 if ok else 0.55
                ax_v.plot(tr.t_s, tr.v_m_s, label=label, alpha=alpha, **style)
                ax_d.plot(tr.t_s, np.degrees(tr.delta_rad), label=label, alpha=alpha, **style)

            ax_v.set_title(f"{title}: Speed")
            ax_v.set_xlabel("t (s)")
            ax_v.set_ylabel("v (m/s)")
            ax_v.grid(True, alpha=0.22, linewidth=0.6)

            ax_d.set_title(f"{title}: Steering")
            ax_d.set_xlabel("t (s)")
            ax_d.set_ylabel("delta (deg)")
            ax_d.grid(True, alpha=0.22, linewidth=0.6)
            ax_d.legend(fontsize=8, loc="best")

        fig.suptitle(str(suptitle))
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    envs_to_plot = list(args.envs)[:4]
    panels: list[tuple[str, int]] = []
    env0_base = str(envs_to_plot[0]).split("::", 1)[0] if envs_to_plot else ""
    multi_pair_fig = (
        bool(getattr(args, "random_start_goal", False))
        and int(len(args.envs)) == 1
        and int(args.runs) >= 4
        and str(env0_base) in set(FOREST_ENV_ORDER)
    )
    if envs_to_plot and multi_pair_fig:
        base = int(getattr(args, "plot_run_idx", 0))
        panels = [(str(envs_to_plot[0]), (base + k) % int(args.runs)) for k in range(4)]
    else:
        for env_name in envs_to_plot:
            env_base = str(env_name).split("::", 1)[0]
            run_idx = 0
            if bool(getattr(args, "random_start_goal", False)) and str(env_base) in set(FOREST_ENV_ORDER):
                run_idx = int(getattr(args, "plot_run_idx", 0))
            panels.append((str(env_name), int(run_idx)))

    fig12_path = out_dir / "fig12_paths.png"
    write_paths_figure(
        panels=panels,
        out_path=fig12_path,
        suptitle="Simulation results of different path-planning methods",
        multi_pair_titles=bool(multi_pair_fig),
    )

    print(f"Wrote: {fig12_path}")

    fig13_path = out_dir / "fig13_controls.png"
    write_controls_figure(
        panels=panels,
        out_path=fig13_path,
        suptitle="Speed and steering of different path-planning methods",
        multi_pair_titles=bool(multi_pair_fig),
    )
    if fig13_path.exists():
        print(f"Wrote: {fig13_path}")

    # Optional: one figure per run index (short + long in the same image).
    if bool(getattr(args, "plot_pair_runs", False)) and bool(getattr(args, "random_start_goal", False)) and bool(
        getattr(args, "rand_two_suites", False)
    ):
        per_run_cap = int(getattr(args, "plot_pair_runs_max", 10))
        per_run_n = int(args.runs)
        if per_run_cap > 0:
            per_run_n = min(int(per_run_n), int(per_run_cap))

        pairs_by_base: dict[str, dict[str, str]] = {}
        for env_name in args.envs:
            env_case = str(env_name)
            if "::" not in env_case:
                continue
            base, suite = env_case.split("::", 1)
            base = str(base).strip()
            suite = str(suite).strip()
            if suite not in {"short", "long"}:
                continue
            pairs_by_base.setdefault(base, {})[suite] = env_case

        for base, suite_map in pairs_by_base.items():
            if "short" not in suite_map or "long" not in suite_map:
                continue
            base_slug = _safe_slug(base)
            for run_idx in range(int(per_run_n)):
                out_path = out_dir / f"fig12_paths_{base_slug}_run_{run_idx:02d}.png"
                write_paths_figure(
                    panels=[(suite_map["short"], int(run_idx)), (suite_map["long"], int(run_idx))],
                    out_path=out_path,
                    suptitle=f"Simulation results of different path-planning methods (run {run_idx})",
                )
                print(f"Wrote: {out_path}")

    print(f"Wrote: {out_dir / 'table2_kpis.csv'}")
    print(f"Wrote: {out_dir / 'table2_kpis_raw.csv'}")
    print(f"Wrote: {out_dir / 'table2_kpis.md'}")
    print(f"Wrote: {out_dir / 'table2_kpis_mean.csv'}")
    print(f"Wrote: {out_dir / 'table2_kpis_mean_raw.csv'}")
    print(f"Wrote: {out_dir / 'table2_kpis_mean.md'}")
    print(f"Run dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
