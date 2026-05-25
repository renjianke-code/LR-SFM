"""Training loop and best-of-K evaluation for LR-SFM.

Trains with AdamW + warm-up cosine + EMA + grad-clip. Evaluates best-of-K
minADE / minFDE / CRPS, with optional KDE-NLL.
"""
import copy
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from .dct import complete_dct_high_modes, fit_dct_high_linear, idct_1d, set_seed
from .ema import EMA


def _kde_nll_batch(pred: torch.Tensor, gt: torch.Tensor, *,
                   log_pdf_lower_bound: float = -20.0) -> torch.Tensor:
    """Per-timestep Gaussian KDE NLL for K trajectory samples.

    ``pred`` is ``[B, K, T, 2]`` and ``gt`` is ``[B, T, 2]``. The return value
    is one NLL per batch item, averaged over future timesteps. This metric is
    evaluation-only and intentionally stays on CPU because SciPy's
    ``gaussian_kde`` is the common reference implementation in this codebase.
    """
    from scipy.stats import gaussian_kde

    pred_np = pred.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()
    B, _, T, _ = pred_np.shape
    out = np.zeros(B, dtype=np.float64)
    for b in range(B):
        ll = 0.0
        for t in range(T):
            try:
                kde = gaussian_kde(pred_np[b, :, t].T)
                log_pdf = kde.logpdf(gt_np[b, t])[0]
            except np.linalg.LinAlgError:
                log_pdf = log_pdf_lower_bound
            ll += max(float(log_pdf), log_pdf_lower_bound)
        out[b] = -ll / T
    return torch.from_numpy(out).to(device=pred.device, dtype=pred.dtype)


def _snapshot_rng():
    state = {
        "py": random.getstate(),
        "np": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state):
    random.setstate(state["py"])
    np.random.set_state(state["np"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _fmt4(x: float) -> str:
    """Format ADE/FDE with enough precision for tuning runs."""
    return f"{x:.4f}"


def _scene_batch(scene_map: torch.Tensor | None,
                 index,
                 device: torch.device) -> torch.Tensor | None:
    if scene_map is None:
        return None
    if isinstance(scene_map, np.ndarray):
        if isinstance(index, slice):
            scene_np = scene_map[index]
        else:
            idx = index.detach().cpu().numpy() if torch.is_tensor(index) else index
            scene_np = scene_map[idx]
        scene = torch.from_numpy(np.array(scene_np, copy=True))
        if scene.dtype == torch.uint8:
            scene = scene.float().div(255.0)
        else:
            scene = scene.float()
        return scene.to(device, non_blocking=True)
    if isinstance(index, slice):
        scene = scene_map[index]
    else:
        idx = index.to(scene_map.device) if scene_map.is_cuda else index.detach().cpu()
        scene = scene_map[idx]
    if scene.dtype == torch.uint8:
        scene = scene.float().div(255.0)
    else:
        scene = scene.float()
    return scene.to(device, non_blocking=True)


def _dct_padding(stats: dict, L: int, dev: torch.device):
    """Return train-only high-DCT padding tensors for the current L."""
    train_mean = stats["train_dct_mean"].to(dev)
    full_frames = int(train_mean.shape[-2])
    mode = stats.get("padding", "mean")
    if mode not in {"mean", "zero", "linear"}:
        raise ValueError(f"Unknown DCT padding mode: {mode}")
    mean_high = None if mode == "zero" else train_mean[L:]

    linear_weight = None
    if mode == "linear" and L < full_frames:
        key = f"dct_high_linear_weight_L{L}"
        if key not in stats:
            full = stats.get("train_dct_full")
            if full is None:
                raise ValueError("padding='linear' requires stats['train_dct_full']")
            full = full.to(dev)
            stats[key] = fit_dct_high_linear(
                full[..., :L, :],
                full[..., L:, :],
                ridge=1e-4,
            ).to(dev)
        linear_weight = stats[key].to(dev)
    return mean_high, linear_weight, full_frames


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train(model, data, stats, *,
          epochs: int = 150,
          bs: int = 256,
          lr: float = 2e-4,
          weight_decay: float = 0.01,
          optimizer: str = "adamw",
          ema_decay: float = 0.995,
          warmup_frac: float = 0.05,
          forward_kwargs: dict,
          eval_fn,
          eval_every: int = 5,
          selection_metric: str = "ade",
          save_path: str | None = None,
          name: str = "model",
          sdl_stop_epoch: int = 0):
    """Train ``model`` with AdamW + warm-up cosine + EMA + grad-clip.

    ``forward_kwargs`` is passed verbatim to ``model.forward`` each step (e.g. the
    DCT min/max buffers and loss weights). ``eval_fn(ema_model)`` should return
    ``"minADE"`` and ``"minFDE"``; by default the best EMA state is kept by lowest
    minADE, with optional FDE-focused selection for tuning.
    """
    if selection_metric not in {"ade", "fde", "ade_fde"}:
        raise ValueError(f"Unknown selection_metric: {selection_metric}")
    train_obs      = data["train_obs"]
    train_dct_norm = data["train_dct_norm"]
    train_nbr_obs  = data["train_nbr_obs"]
    train_nbr_mask = data["train_nbr_mask"]
    N = len(train_obs)
    train_scene_map = data.get("train_scene_map")                # [N, G, G] or None
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if bs < 1:
        raise ValueError("bs must be at least 1")
    if sdl_stop_epoch < 0:
        raise ValueError("sdl_stop_epoch must be non-negative")
    if N < 1:
        raise ValueError("training data is empty")
    iters_per_epoch = (N + bs - 1) // bs
    total_iters = epochs * iters_per_epoch
    warmup_iters = min(total_iters, max(1, int(total_iters * warmup_frac)))

    if optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    warmup = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(1e-6 / lr, s / warmup_iters))
    if total_iters > warmup_iters:
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_iters - warmup_iters, eta_min=1e-6)
        sched = torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[warmup_iters])
    else:
        sched = warmup

    ema = EMA(model, decay=ema_decay)
    best_score = float("inf")
    best_state = None

    # Reshape train_dct_norm to model's flat input shape
    if train_dct_norm.dim() == 4:                 # [N, A, L, 2]
        train_x1 = train_dct_norm[:, :, :model.L, :].reshape(N, model.A, model.out_dim)
    else:                                          # [N, L, 2]
        train_x1 = train_dct_norm[:, :model.L, :].reshape(N, model.out_dim)

    forward_kwargs = forward_kwargs.copy()
    if float(forward_kwargs.get("sdl_std_pull_weight", 0.0)) > 0:
        if train_dct_norm.dim() == 4:
            train_modes = train_dct_norm[:, :, :model.L, :]
        else:
            train_modes = train_dct_norm[:, None, :model.L, :]
        forward_kwargs["sdl_mode_std"] = train_modes.std(
            dim=(0, 1, 3), unbiased=False,
        ).clamp_min(1e-6).detach()

    pbar = tqdm(range(1, epochs + 1), desc=name, dynamic_ncols=True)
    best_ade = best_fde = float("nan")
    sdl_scheduled_keys = (
        "contrastive_weight",
        "sdl_std_pull_weight",
    )
    base_sdl_weights = {
        key: float(forward_kwargs.get(key, 0.0))
        for key in sdl_scheduled_keys
    }
    has_sdl_schedule = sdl_stop_epoch > 0
    for ep in pbar:
        model.train()
        tot, tot_reg, n = 0.0, 0.0, 0
        order = torch.randperm(N, device=train_obs.device)
        ep_forward_kwargs = forward_kwargs
        if has_sdl_schedule and any(v != 0.0 for v in base_sdl_weights.values()):
            sdl_scale = 1.0
            if sdl_stop_epoch > 0 and ep > sdl_stop_epoch:
                sdl_scale = 0.0
            ep_forward_kwargs = forward_kwargs.copy()
            for key, value in base_sdl_weights.items():
                ep_forward_kwargs[key] = value * sdl_scale
        for start in range(0, N, bs):
            bi = order[start:start + bs]
            obs_b = train_obs[bi]
            x1_b  = train_x1[bi]
            scene_b = _scene_batch(train_scene_map, bi, obs_b.device)
            opt.zero_grad()
            total, reg, _ = model(
                obs_b, x1_b,
                neighbor_obs=train_nbr_obs[bi] if train_nbr_obs is not None else None,
                neighbor_mask=train_nbr_mask[bi] if train_nbr_mask is not None else None,
                dct_min=stats["minmax_min"],
                dct_max=stats["minmax_max"],
                scene_heatmap=scene_b,
                **ep_forward_kwargs,
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); ema.update(model)
            tot += total.item(); tot_reg += reg; n += 1

        if ep % eval_every == 0 or ep == epochs:
            res = eval_fn(ema.shadow)
            if selection_metric == "fde":
                score = res["minFDE"]
            elif selection_metric == "ade_fde":
                score = res["minADE"] + res["minFDE"]
            else:
                score = res["minADE"]
            if score < best_score:
                best_score = score
                best_ade, best_fde = res["minADE"], res["minFDE"]
                best_state = copy.deepcopy(ema.state_dict())
                if save_path is not None:
                    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                    torch.save({"ema": best_state, "epoch": ep}, save_path)

        pbar.set_postfix(
            ADE=_fmt4(best_ade) if best_ade == best_ade else "—",
            FDE=_fmt4(best_fde) if best_fde == best_fde else "—",
            loss=f"{tot/n:.3f}",
        )

    if best_state is not None:
        ema.load_state_dict(best_state)
    return ema.shadow


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, data, stats, *, K: int = 20, steps: int = 3, batch_size: int = 64,
             solver: str = "euler", lin_poly_p: int = 5, lin_poly_long_step: int = 1000,
             tied_noise: bool = True, compute_kde_nll: bool = False):
    """Best-of-K minADE / minFDE + CRPS.

    Works for both A=1 (ETH-UCY, SDD) and A>1 (NBA) thanks to the unified
    ``data`` schema in :mod:`lr_sfm.data`. CRPS = meanADE - 0.5 * APD_ade is
    used as the training-time best-checkpoint selection criterion (it rewards
    accurate-and-diverse predictions, not just the single best hypothesis).
    """
    rng_state = _snapshot_rng()
    try:
        set_seed(42)
        return _evaluate_inner(
            model, data, stats, K=K, steps=steps, batch_size=batch_size,
            solver=solver, lin_poly_p=lin_poly_p,
            lin_poly_long_step=lin_poly_long_step, tied_noise=tied_noise,
            compute_kde_nll=compute_kde_nll,
        )
    finally:
        _restore_rng(rng_state)


def _evaluate_inner(model, data, stats, *, K, steps, batch_size,
                    solver, lin_poly_p, lin_poly_long_step, tied_noise,
                    compute_kde_nll):
    model.eval()

    test_obs      = data["test_obs"]
    test_pred     = data["test_pred"]
    test_nbr_obs  = data["test_nbr_obs"]
    test_nbr_mask = data["test_nbr_mask"]
    test_rot_inv  = data.get("test_rot_inv")
    test_scale    = data.get("test_scale")
    test_last_obs = data["test_last_obs"]
    recover_mode  = data.get("test_recover_mode", "relative_last")
    N = len(test_obs)

    mn, mx = stats["minmax_min"], stats["minmax_max"]
    L = model.L
    train_dct_mean_high, dct_high_linear_weight, dct_full_frames = _dct_padding(
        stats, L, test_obs.device,
    )
    test_scene_map = data.get("test_scene_map")

    all_min_ade, all_min_fde, all_mean_ade, all_apd, all_kde_nll = [], [], [], [], []
    for s in range(0, N, batch_size):
        e = s + batch_size
        obs   = test_obs[s:e]
        fut   = test_pred[s:e]
        nbr   = test_nbr_obs[s:e] if test_nbr_obs is not None else None
        nbrm  = test_nbr_mask[s:e] if test_nbr_mask is not None else None
        last  = test_last_obs[s:e]
        scene = _scene_batch(test_scene_map, slice(s, e), obs.device)
        B = obs.shape[0]

        samp, _ = model.sample(
            obs, K=K, steps=steps,
            neighbor_obs=nbr, neighbor_mask=nbrm,
            scene_heatmap=scene,
            solver=solver,
            lin_poly_p=lin_poly_p,
            lin_poly_long_step=lin_poly_long_step,
            tied_noise=tied_noise,
        )
        # samp: [B, K, out_dim] (A=1) or [B, K, A, out_dim]
        if samp.dim() == 4:                                   # multi-agent (NBA)
            A = samp.shape[2]
            BKA = B * K * A
            coarse = samp.reshape(BKA, L, 2)
            coarse = (coarse + 1.0) / 2.0 * (mx[:L] - mn[:L]) + mn[:L]
            full = complete_dct_high_modes(
                coarse,
                mean_high=train_dct_mean_high,
                linear_weight=dct_high_linear_weight,
                full_frames=dct_full_frames,
            )
            T = full.shape[1]
            pred_rel = idct_1d(full).reshape(B, K, A, T, 2)
            pred = pred_rel + last[:, None]                   # [B, K, A, T, 2]
            err = (pred - fut[:, None]).norm(dim=-1)          # [B, K, A, T]
            ade_per_k = err.mean(-1)                          # [B, K, A]
            min_ade = ade_per_k.min(1)[0].mean(-1)            # [B], avg over agents
            min_fde = err[..., -1].min(1)[0].mean(-1)
            mean_ade = ade_per_k.mean(1).mean(-1)             # [B]
            # APD: pairwise distance between K hypotheses, averaged over time
            diff = pred.unsqueeze(2) - pred.unsqueeze(1)      # [B, K, K, A, T, 2]
            pd = diff.norm(dim=-1).mean(dim=-1)               # [B, K, K, A]
            eye = torch.eye(K, dtype=torch.bool, device=pred.device)
            apd = pd[:, ~eye].reshape(B, K * (K - 1), A).mean(dim=(1, 2))   # [B]
            if compute_kde_nll:
                kde_nll = _kde_nll_batch(
                    pred.permute(0, 2, 1, 3, 4).reshape(B * A, K, T, 2),
                    fut.reshape(B * A, T, 2),
                ).reshape(B, A).mean(dim=1)
        else:                                                  # A=1
            coarse = samp.reshape(B * K, L, 2)
            coarse = (coarse + 1.0) / 2.0 * (mx[:L] - mn[:L]) + mn[:L]
            full = complete_dct_high_modes(
                coarse,
                mean_high=train_dct_mean_high,
                linear_weight=dct_high_linear_weight,
                full_frames=dct_full_frames,
            )
            T = full.shape[1]
            pred_rel = idct_1d(full).reshape(B, K, T, 2)  # in rotated frame
            if test_scale is not None:
                scale = test_scale[s:e].to(pred_rel.device).view(B, 1, 1, 1)
                if recover_mode == "scaled_absolute":
                    first = pred_rel[:, :, :1, :]
                    pred_rel = (pred_rel - first) * scale + first
                else:
                    pred_rel = pred_rel * scale
            if test_rot_inv is not None:
                rot_inv = test_rot_inv[s:e]
                pred_rel = torch.einsum("bktc,bcd->bktd", pred_rel, rot_inv)
            pred = pred_rel + last.unsqueeze(1)
            err = (pred - fut.unsqueeze(1)).norm(dim=-1)      # [B, K, T]
            ade_per_k = err.mean(-1)                          # [B, K]
            min_ade = ade_per_k.min(1)[0]
            min_fde = err[..., -1].min(1)[0]
            mean_ade = ade_per_k.mean(1)
            diff = pred.unsqueeze(2) - pred.unsqueeze(1)      # [B, K, K, T, 2]
            pd = diff.norm(dim=-1).mean(dim=-1)               # [B, K, K]
            eye = torch.eye(K, dtype=torch.bool, device=pred.device)
            apd = pd[:, ~eye].reshape(B, K * (K - 1)).mean(dim=1)
            if compute_kde_nll:
                kde_nll = _kde_nll_batch(pred, fut)

        all_min_ade.append(min_ade)
        all_min_fde.append(min_fde)
        all_mean_ade.append(mean_ade)
        all_apd.append(apd)
        if compute_kde_nll:
            all_kde_nll.append(kde_nll)

    mean_ade_all = torch.cat(all_mean_ade).mean().item()
    apd_all      = torch.cat(all_apd).mean().item()
    out = {
        "minADE":  torch.cat(all_min_ade).mean().item(),
        "minFDE":  torch.cat(all_min_fde).mean().item(),
        "meanADE": mean_ade_all,
        "APD":     apd_all,
        "CRPS":    mean_ade_all - 0.5 * apd_all,
    }
    if compute_kde_nll:
        out["KDE_NLL"] = torch.cat(all_kde_nll).mean().item()
    return out
