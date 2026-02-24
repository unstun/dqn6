"""Shared forest action-selection pipeline used by both train and infer.

V7: eliminates train/infer mismatch by running the *exact same* gating
logic (greedy Q → admissible check → top-k → prog_mask → fallback) in
both contexts.  The only difference is the ``explore`` flag which enables
epsilon-greedy exploration during training.
"""

from __future__ import annotations

import numpy as np
import torch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from amr_dqn.agents import DQNFamilyAgent
    from amr_dqn.env import AMRBicycleEnv


def forest_select_action(
    env: AMRBicycleEnv,
    agent: DQNFamilyAgent,
    obs: np.ndarray,
    *,
    episode: int,
    explore: bool,
    horizon_steps: int = 15,
    topk: int = 10,
    min_od_m: float = 0.0,
    min_progress_m: float = 1e-4,
) -> int:
    """Unified action selection for AMRBicycleEnv (forest) environments.

    Pipeline (identical for train and infer):
        1. Compute Q values
        2. If *explore* and epsilon fires → random choice from admissible mask
        3. Greedy a0 = argmax Q
        4. If a0 is admissible → return a0
        5. Top-k search: try next-best Q actions for admissibility
        6. prog_mask fallback: masked argmax Q over admissible actions
        7. Last resort: env heuristic short-rollout fallback
    """
    adm_h = max(1, int(horizon_steps))
    topk_k = max(1, int(topk))
    min_od = float(min_od_m)
    min_prog = float(min_progress_m)

    # --- epsilon exploration (training only) ---------------------------------
    if explore and (agent._rng.random() < agent.epsilon(episode)):
        mask = env.admissible_action_mask(
            horizon_steps=adm_h,
            min_od_m=min_od,
            min_progress_m=min_prog,
            fallback_to_safe=True,
        )
        idxs = np.nonzero(mask)[0]
        if idxs.size == 0:
            return int(agent._rng.integers(0, agent._n_actions))
        return int(agent._rng.choice(idxs))

    # --- greedy path (shared by train and infer) -----------------------------
    with torch.no_grad():
        x = torch.from_numpy(agent._prep_obs(obs)).to(agent.device)
        q = agent.q(x.unsqueeze(0)).squeeze(0)

    a0 = int(torch.argmax(q).item())

    # Step 4: admissibility check on greedy action
    if bool(env.is_action_admissible(int(a0), horizon_steps=adm_h, min_od_m=min_od, min_progress_m=min_prog)):
        return int(a0)

    # Step 5: top-k search
    kk = int(min(topk_k, int(q.numel())))
    topk_indices = torch.topk(q, k=kk, dim=0).indices.detach().cpu().numpy()
    for cand in topk_indices.tolist():
        cand_i = int(cand)
        if cand_i == int(a0):
            continue
        if bool(env.is_action_admissible(cand_i, horizon_steps=adm_h, min_od_m=min_od, min_progress_m=min_prog)):
            return int(cand_i)

    # Step 6: prog_mask fallback (masked argmax Q)
    prog_mask = env.admissible_action_mask(
        horizon_steps=adm_h,
        min_od_m=min_od,
        min_progress_m=min_prog,
        fallback_to_safe=False,
    )
    if bool(prog_mask.any()):
        q_masked = q.clone()
        q_masked[torch.from_numpy(~prog_mask).to(q.device)] = torch.finfo(q_masked.dtype).min
        return int(torch.argmax(q_masked).item())

    # Step 7: heuristic fallback
    return int(env._fallback_action_short_rollout(horizon_steps=adm_h, min_od_m=min_od))


def forest_compute_next_mask(
    env: AMRBicycleEnv,
    *,
    horizon_steps: int = 15,
    min_od_m: float = 0.0,
    min_progress_m: float = 1e-4,
) -> np.ndarray:
    """Compute admissible-action mask for the *next* state (replay buffer TD target)."""
    return env.admissible_action_mask(
        horizon_steps=max(1, int(horizon_steps)),
        min_od_m=float(min_od_m),
        min_progress_m=float(min_progress_m),
        fallback_to_safe=True,
    )
