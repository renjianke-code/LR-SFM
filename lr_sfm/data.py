"""Dataset loading and preprocessing for ETH-UCY, SDD, and NBA.

Each ``prepare_*`` returns a ``(data, stats)`` tuple ready for training and
evaluation. Conventions for the ``data`` dict:

    train_obs        - encoder input  [N, T_obs, D_in]  (D_in=2 for ETH/SDD, 6 for NBA)
    train_dct_norm   - DCT(future) min-max normalised  [N, k, 2] or [N, A, k, 2]
    train_nbr_obs    - encoder neighbour input
    train_nbr_mask   - neighbour validity mask
    test_obs / test_pred / test_nbr_obs / test_nbr_mask  - same for the held-out split
    test_rot_inv     - rotation matrix that maps predictions back to world coords
                       (identity for NBA since no alignment is applied)
    test_scale       - optional scale factor to undo trajectory-standard normalisation
    test_last_obs    - reference absolute position to add after iDCT

``stats`` carries the DCT min/max used for normalisation and the high-freq DCT mean
used to pad discarded modes during inference.
"""
import math
import os
import plistlib

import numpy as np
import torch

from .dct import dct_1d, device


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_STANDARD_ROOT = os.path.join(_REPO_ROOT, "data", "trajectory_standard")


def _resolve_standard_root(data_root: str | None, split_name: str) -> str:
    """Find the standard trajectory data root for ETH-UCY or SDD."""
    candidates = []
    if data_root:
        candidates.extend([
            data_root,
            os.path.join(data_root, "trajectory_standard"),
            os.path.join(data_root, "standard"),
        ])
    candidates.extend([
        _DEFAULT_STANDARD_ROOT,
    ])

    seen = set()
    for root in candidates:
        root = os.path.abspath(root)
        if root in seen:
            continue
        seen.add(root)
        split_path = os.path.join(root, "datasets", f"{split_name}.plist")
        if not os.path.exists(split_path):
            continue
        with open(split_path, "rb") as f:
            split = plistlib.load(f)
        names = split.get("train", []) + split.get("test", [])
        if not names:
            continue
        subset_path = os.path.join(root, "datasets", "subsets", f"{names[0]}.plist")
        if not os.path.exists(subset_path):
            continue
        with open(subset_path, "rb") as f:
            subset = plistlib.load(f)
        csv_path = os.path.join(root, subset["dataset_dir"], "true_pos_.csv")
        if os.path.exists(csv_path):
            return root
    raise FileNotFoundError(
        f"Could not find standard trajectory data for split '{split_name}'. "
        "Expected datasets/*.plist plus data/*/true_pos_.csv."
    )


def _standard_cache_root(data_root: str | None) -> str:
    return os.path.abspath(data_root or _DEFAULT_STANDARD_ROOT)


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------
def _minmax_normalise(dct: torch.Tensor):
    """Per-mode min-max normalise DCT coefficients to [-1, 1]."""
    flat = dct.reshape(-1, *dct.shape[-2:])
    mn = flat.min(0).values
    mx = flat.max(0).values
    denom = (mx - mn).clamp_min(1e-6)
    normed = 2.0 * (dct - mn) / denom - 1.0
    return normed, mn, mx


def _heading_rotate_sdd(obs_rel: torch.Tensor, pred_rel: torch.Tensor,
                   nbr_rel: torch.Tensor, t_frame: int = 6):
    """Heading alignment for SDD: rotate by direction of obs_rel[t_frame]."""
    disp = obs_rel[:, t_frame, :]
    theta = torch.atan2(disp[:, 1], disp[:, 0] + 1e-5)
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    rot = torch.zeros(len(theta), 2, 2, device=obs_rel.device)
    rot[:, 0, 0] = cos_t;  rot[:, 0, 1] = sin_t
    rot[:, 1, 0] = -sin_t; rot[:, 1, 1] = cos_t
    obs_rot = torch.einsum("bij,btj->bti", rot, obs_rel)
    pred_rot = torch.einsum("bij,btj->bti", rot, pred_rel)
    N, M, T, _ = nbr_rel.shape
    if M == 0:
        nbr_rot = nbr_rel.clone()
    else:
        nbr_rot = torch.einsum("bij,btj->bti", rot, nbr_rel.reshape(N, M * T, 2)).reshape(N, M, T, 2)
    # row-vec rot_inv: applying as ``pred @ rot_inv`` un-rotates back to world frame.
    return obs_rot, pred_rot, nbr_rot, rot


def _standard_subset_samples(standard_root: str,
                          subset: str,
                          *,
                          obs_len: int,
                          pred_len: int,
                          step: int = 1,
                          init_position: float = 10000.0) -> torch.Tensor:
    """Sample one standard trajectory subset from ``true_pos_.csv``.

    This is a clean-room implementation of the public standard trajectory protocol:
    read subset metadata from ``datasets/subsets/*.plist``, build a dense
    frame/person matrix from ``true_pos_.csv``, and slide one frame at a time
    over each person's visible segment.
    """
    plist_path = os.path.join(standard_root, "datasets", "subsets", f"{subset}.plist")
    with open(plist_path, "rb") as f:
        info = plistlib.load(f)

    csv_path = os.path.join(standard_root, info["dataset_dir"], "true_pos_.csv")
    rows = np.genfromtxt(csv_path, delimiter=",").T.astype(np.float32)
    order = [int(i) for i in info["order"]]
    sample_rate, frame_rate = [float(i) for i in info["paras"]]
    frame_step = max(1, int(0.4 / (sample_rate / frame_rate)))

    frames = sorted(set(rows[:, 0].astype(np.int32).tolist()))
    ped_ids = sorted(set(rows[:, 1].astype(np.int32).tolist()))
    frame_to_i = {fr: i for i, fr in enumerate(frames)}
    ped_to_i = {pid: i for i, pid in enumerate(ped_ids)}
    matrix = init_position * np.ones((len(frames), len(ped_ids), 2), dtype=np.float32)
    for row in rows:
        matrix[frame_to_i[int(row[0])], ped_to_i[int(row[1])]] = [
            row[2 + order[0]], row[2 + order[1]],
        ]

    seq_len = obs_len + pred_len
    samples = []
    for ped_i in range(matrix.shape[1]):
        base = matrix[:, ped_i, 0]
        diff = base[:-1] - base[1:]
        appear = np.where(diff > init_position / 2)[0]
        disappear = np.where(diff < -init_position / 2)[0]
        start = int(appear[0] + 1) if len(appear) else 0
        end = int(disappear[0] + 1) if len(disappear) else len(base)
        for p in range(start, end, step * frame_step):
            e = p + seq_len * frame_step
            if e > end:
                break
            samples.append(matrix[p:e:frame_step, ped_i].copy())

    if not samples:
        return torch.empty(0, obs_len + pred_len, 2)
    return torch.from_numpy(np.stack(samples, axis=0)).float()


def _standard_protocol_split(standard_root: str,
                          test_scene: str,
                          *,
                          obs_len: int,
                          pred_len: int):
    plist_path = os.path.join(standard_root, "datasets", f"{test_scene}.plist")
    with open(plist_path, "rb") as f:
        split = plistlib.load(f)

    def _load_subsets(names):
        chunks = [
            _standard_subset_samples(
                standard_root, name,
                obs_len=obs_len, pred_len=pred_len,
            )
            for name in names
        ]
        return torch.cat(chunks, dim=0) if chunks else torch.empty(0, obs_len + pred_len, 2)

    train = _load_subsets(split["train"])
    test = _load_subsets(split["test"])
    return train[:, :obs_len], train[:, obs_len:], test[:, :obs_len], test[:, obs_len:]


def _standard_subset_matrix(standard_root: str,
                         subset: str,
                         *,
                         init_position: float = 10000.0):
    plist_path = os.path.join(standard_root, "datasets", "subsets", f"{subset}.plist")
    with open(plist_path, "rb") as f:
        info = plistlib.load(f)

    csv_path = os.path.join(standard_root, info["dataset_dir"], "true_pos_.csv")
    rows = np.genfromtxt(csv_path, delimiter=",").T.astype(np.float32)
    order = [int(i) for i in info["order"]]
    sample_rate, frame_rate = [float(i) for i in info["paras"]]
    frame_step = max(1, int(0.4 / (sample_rate / frame_rate)))

    frames = sorted(set(rows[:, 0].astype(np.int32).tolist()))
    ped_ids = sorted(set(rows[:, 1].astype(np.int32).tolist()))
    frame_to_i = {fr: i for i, fr in enumerate(frames)}
    ped_to_i = {pid: i for i, pid in enumerate(ped_ids)}
    matrix = init_position * np.ones((len(frames), len(ped_ids), 2), dtype=np.float32)
    for row in rows:
        matrix[frame_to_i[int(row[0])], ped_to_i[int(row[1])]] = [
            row[2 + order[0]], row[2 + order[1]],
        ]
    neighbors = [
        np.where(np.not_equal(frame[:, 0], init_position))[0]
        for frame in matrix
    ]
    return matrix, frames, neighbors, frame_step


def _standard_subset_sample_records(standard_root: str,
                                 subset: str,
                                 *,
                                 obs_len: int,
                                 pred_len: int,
                                 step: int = 1,
                                 init_position: float = 10000.0,
                                 max_neighbors: int = 15):
    matrix, frames, neighbors, frame_step = _standard_subset_matrix(
        standard_root, subset, init_position=init_position,
    )
    seq_len = obs_len + pred_len
    records = []
    for ped_i in range(matrix.shape[1]):
        base = matrix[:, ped_i, 0]
        diff = base[:-1] - base[1:]
        appear = np.where(diff > init_position / 2)[0]
        disappear = np.where(diff < -init_position / 2)[0]
        start = int(appear[0] + 1) if len(appear) else 0
        end = int(disappear[0] + 1) if len(disappear) else len(base)
        for p in range(start, end, step * frame_step):
            obs_frame = p + obs_len * frame_step
            e = p + seq_len * frame_step
            if e > end:
                break

            full = matrix[p:e:frame_step, ped_i].copy()
            present = np.asarray(neighbors[obs_frame - frame_step], dtype=np.int64)
            if len(present) > max_neighbors + 1:
                target_pos = matrix[obs_frame - frame_step, ped_i][None, :]
                dist = _traj_length(matrix[obs_frame - frame_step, present] - target_pos)
                present = present[np.argsort(dist)[1:max_neighbors + 1]]
            neighbor_trajs = [
                matrix[p:e:frame_step, n].astype(np.float32).copy()
                for n in present
            ]
            records.append((full[:obs_len].astype(np.float32), neighbor_trajs))
    return records


def _build_trajectory_guidance_map(guidance_obs: np.ndarray,
                              *,
                              standard_root: str,
                              window_size: float = 10.0,
                              expand_meter: float = 10.0):
    flat = guidance_obs.reshape(-1, 2)
    x_min, y_min = flat.min(axis=0)
    x_max, y_max = flat.max(axis=0)
    # Keep these real-to-grid parameters in NumPy's default float precision.
    # Using float32 shifts a few centers by one grid cell at int
    # truncation boundaries.
    W = np.array([window_size, window_size])
    b = np.array([x_min - expand_meter, y_min - expand_meter])
    guidance_shape = (
        int((x_max - x_min + 2 * expand_meter) * W[0]) + 1,
        int((y_max - y_min + 2 * expand_meter) * W[1]) + 1,
    )
    guidance = np.zeros(guidance_shape, dtype=np.float32)
    guidance = _stamp_trajectories(
        guidance, _trajectory_real2grid(guidance_obs, W, b),
        amplitudes=1.0, radii=7, decay=False, clip=False, mask_root=standard_root,
    )
    guidance = np.minimum(guidance, 30.0)
    gmax = guidance.max()
    guidance = 1.0 - guidance / gmax if gmax > 1e-6 else np.ones_like(guidance)
    return guidance.astype(np.float32), W, b


def _build_social_map_full(target_obs: np.ndarray,
                                 neighbor_trajs: list[np.ndarray],
                                 *,
                                 W: np.ndarray,
                                 b: np.ndarray,
                                 shape: tuple[int, int],
                                 obs_len: int,
                                 pred_len: int,
                                 max_neighbors: int = 15,
                                 standard_root: str | None = None) -> np.ndarray:
    total_len = obs_len + pred_len
    target_pred = _linear_predict_path(target_obs, total_len)[obs_len:]

    neighbor_preds = []
    for traj in neighbor_trajs:
        filled = _fill_missing_trajectory(traj)
        if filled is None:
            continue
        neighbor_preds.append(_linear_predict_path(filled, total_len)[obs_len:])

    trajs = [target_pred]
    amps = [-2.0]
    radii = [20]
    if neighbor_preds:
        neighbor_preds_np = np.asarray(neighbor_preds, dtype=np.float32)
        vec_target = target_pred[-1] - target_pred[0]
        len_target = _traj_length(vec_target)
        vec_neighbor = neighbor_preds_np[:, -1] - neighbor_preds_np[:, 0]
        if len_target >= 0.05:
            cosine = _activation_np(_cosine_np(vec_target[None, :], vec_neighbor), a=1.0, b=0.2)
            velocity = _traj_length(vec_neighbor) / len_target
        else:
            cosine = np.ones(len(neighbor_preds_np), dtype=np.float32)
            velocity = np.full(len(neighbor_preds_np), 2.0, dtype=np.float32)
        amps.extend((-cosine * velocity).tolist())
        radii.extend([15] * len(neighbor_preds_np))
        trajs.extend(neighbor_preds_np.tolist())

    trajs = np.asarray(trajs, dtype=np.float32)
    if len(trajs) > max_neighbors + 1:
        dist = _traj_length(trajs[:1, 0, :] - trajs[:, 0, :])
        keep = np.argsort(dist)[:max_neighbors + 1]
        trajs = trajs[keep]
        # Preserve the reference sampler's quirk: only ``trajs`` are reordered
        # here, while amplitudes/radii remain in their original order.

    source = np.zeros(shape, dtype=np.float32)
    source = _stamp_trajectories(
        source, _trajectory_real2grid(trajs, W, b),
        amplitudes=np.asarray(amps, dtype=np.float32),
        radii=np.asarray(radii, dtype=np.int32),
        decay=True,
        clip=False,
        mask_root=standard_root,
    )
    span = source.max() - source.min()
    if span <= 0.01:
        return 0.5 * np.ones_like(source)
    return (source - source.min()) / span


def _subset_context_maps(standard_root: str,
                               subset: str,
                               *,
                               obs_len: int = 8,
                               pred_len: int = 12,
                               grid: int = 100,
                               max_neighbors: int = 15) -> np.ndarray:
    records = _standard_subset_sample_records(
        standard_root, subset, obs_len=obs_len, pred_len=pred_len,
        max_neighbors=max_neighbors,
    )
    if not records:
        return np.empty((0, grid, grid), dtype=np.uint8)

    guidance_obs = np.asarray([obs for obs, _ in records], dtype=np.float32)
    guidance, W, b = _build_trajectory_guidance_map(guidance_obs, standard_root=standard_root)
    half = grid // 2
    out = np.empty((len(records), grid, grid), dtype=np.uint8)
    for i, (obs, neighbor_trajs) in enumerate(records):
        center_grid = _trajectory_real2grid(obs[-1], W, b)
        traj_crop = _trajectory_cut_one(guidance, center_grid, half)
        social_map = _build_social_map_full(
            obs, neighbor_trajs,
            W=W, b=b, shape=guidance.shape,
            obs_len=obs_len, pred_len=pred_len,
            max_neighbors=max_neighbors,
            standard_root=standard_root,
        )
        social_crop = _trajectory_cut_one(social_map, center_grid, half)
        fused = np.clip(0.5 * traj_crop + 0.5 * social_crop, 0.0, 1.0)
        out[i] = (fused * 255.0).astype(np.uint8)
    return out


def _subset_context_maps_cached(standard_root: str,
                                      subset: str,
                                      *,
                                      obs_len: int,
                                      pred_len: int,
                                      grid: int,
                                      cache_dir: str | None = None) -> np.ndarray:
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(
            cache_dir, f"{subset}_g{grid}_o{obs_len}_p{pred_len}_std.npy",
        )
        if os.path.exists(cache_path):
            cached = np.load(cache_path, mmap_mode="r")
            if cached.shape[1:] == (grid, grid):
                return cached
        maps = _subset_context_maps(
            standard_root, subset, obs_len=obs_len, pred_len=pred_len, grid=grid,
        )
        tmp_path = cache_path + ".tmp"
        with open(tmp_path, "wb") as f:
            np.save(f, maps)
        os.replace(tmp_path, cache_path)
        return np.load(cache_path, mmap_mode="r")

    return _subset_context_maps(
        standard_root, subset, obs_len=obs_len, pred_len=pred_len, grid=grid,
    )


def _standard_context_maps(standard_root: str,
                                 test_scene: str,
                                 split_name: str,
                                 *,
                                 obs_len: int = 8,
                                 pred_len: int = 12,
                                 grid: int = 100,
                                 expected_count: int | None = None,
                                 cache_path: str | None = None):
    plist_path = os.path.join(standard_root, "datasets", f"{test_scene}.plist")
    with open(plist_path, "rb") as f:
        split = plistlib.load(f)
    subset_names = split[split_name]
    subset_cache_dir = None
    if cache_path is not None:
        subset_cache_dir = os.path.join(os.path.dirname(cache_path), "subsets")

    if cache_path is not None and os.path.exists(cache_path):
        cached = np.load(cache_path, mmap_mode="r")
        if cached.shape[1:] == (grid, grid) and (expected_count is None or cached.shape[0] == expected_count):
            return cached

    if expected_count is not None and cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        write_path = cache_path + ".tmp"
        if os.path.exists(write_path):
            os.remove(write_path)
        out = np.lib.format.open_memmap(
            write_path, mode="w+", dtype=np.uint8,
            shape=(expected_count, grid, grid),
        )
        offset = 0
        for name in subset_names:
            maps = _subset_context_maps_cached(
                standard_root, name, obs_len=obs_len, pred_len=pred_len, grid=grid,
                cache_dir=subset_cache_dir,
            )
            out[offset:offset + len(maps)] = maps
            offset += len(maps)
        if offset != expected_count:
            raise ValueError(
                f"standard trajectory protocol context map count mismatch for {test_scene}/{split_name}: "
                f"{offset}/{expected_count}"
            )
        out.flush()
        del out
        os.replace(write_path, cache_path)
        return np.load(cache_path, mmap_mode="r")

    chunks = [
        _subset_context_maps_cached(
            standard_root, name, obs_len=obs_len, pred_len=pred_len, grid=grid,
            cache_dir=subset_cache_dir,
        )
        for name in subset_names
    ]
    out = np.concatenate(chunks, axis=0) if chunks else np.empty((0, grid, grid), dtype=np.uint8)
    if expected_count is not None and len(out) != expected_count:
        raise ValueError(
            f"standard trajectory protocol context map count mismatch for {test_scene}/{split_name}: "
            f"{len(out)}/{expected_count}"
        )
    return torch.from_numpy(out)


def _heading_rotate_eth(past_rel, fut_rel, past_abs, rotate_time_frame=6):
    """Heading alignment for ETH-UCY.

    Rotates so that ``past_rel[:, rotate_time_frame]`` (= ``obs[t]-obs[-1]``)
    aligns with the +x axis. Returns the rotated past_rel, fut_rel, past_abs
    and the rotation matrix needed to undo the rotation on predictions.
    """
    past_diff = past_rel[:, rotate_time_frame]                    # [N, 2]
    theta = torch.atan(past_diff[:, 1] / (past_diff[:, 0] + 1e-5))
    theta = torch.where(past_diff[:, 0] < 0, theta + math.pi, theta)
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    rot = torch.zeros(len(theta), 2, 2, device=past_rel.device)
    rot[:, 0, 0] =  cos_t;  rot[:, 0, 1] = sin_t
    rot[:, 1, 0] = -sin_t;  rot[:, 1, 1] = cos_t
    past_rel_r = torch.einsum("bij,btj->bti", rot, past_rel)
    fut_rel_r  = torch.einsum("bij,btj->bti", rot, fut_rel)
    past_abs_r = torch.einsum("bij,btj->bti", rot, past_abs)
    return past_rel_r, fut_rel_r, past_abs_r, rot


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    ex = np.exp(x)
    return (ex / ex.sum()).astype(np.float32)


def _linear_predict_path(position: np.ndarray,
                          time_pred: int,
                          different_weights: float = 0.95) -> np.ndarray:
    """Weighted linear extrapolation used by standard trajectory context maps."""
    position = np.asarray(position, dtype=np.float32)
    time_obv = position.shape[0]
    t = np.arange(time_obv, dtype=np.float32)
    t_p = np.arange(time_pred, dtype=np.float32)
    if different_weights == 0:
        weights = np.ones(time_obv, dtype=np.float32)
        weights = weights / weights.sum()
    else:
        weights = _softmax_np([(i + 1) ** different_weights for i in range(time_obv)])
    A = np.stack([np.ones_like(t), t], axis=1)
    A_p = np.stack([np.ones_like(t_p), t_p], axis=1)
    Aw = A * weights[:, None]
    coef = np.linalg.pinv(A.T @ Aw) @ A.T @ (weights[:, None] * position)
    return (A_p @ coef).astype(np.float32)


def _traj_length(vec: np.ndarray) -> np.ndarray:
    return np.linalg.norm(vec, axis=-1)


def _cosine_np(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    length1 = _traj_length(vec1)
    length2 = _traj_length(vec2)
    return (np.sum(vec1 * vec2, axis=-1) + 1e-4) / ((length1 * length2) + 1e-4)


def _activation_np(x: np.ndarray, a: float = 1.0, b: float = 1.0) -> np.ndarray:
    return (x <= 0) * a * x + (x > 0) * b * x


def _trajectory_real2grid(traj: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((np.asarray(traj, dtype=np.float32) - b) * W).astype(np.int32)


_CIRCLE_MASK_CACHE = {}


def _circle_mask_kernel(radius: int, mask_root: str | None = None) -> np.ndarray:
    """Standard trajectory-style resized circle mask."""
    key = (int(radius), os.path.abspath(mask_root) if mask_root else None)
    if key in _CIRCLE_MASK_CACHE:
        return _CIRCLE_MASK_CACHE[key]

    if mask_root:
        mask_path = os.path.join(mask_root, "figures", "mask_circle.png")
        if os.path.exists(mask_path):
            try:
                import cv2  # optional; used only when the reference mask exists
                mask = cv2.imread(mask_path)[:, :, 0].astype(np.float32) / 50.0
                resized = cv2.resize(mask, (radius * 2 + 1, radius * 2 + 1))
                _CIRCLE_MASK_CACHE[key] = resized.astype(np.float32)
                return _CIRCLE_MASK_CACHE[key]
            except Exception:
                pass

    size = radius * 2 + 1
    src_size = 101.0
    coords = (np.arange(size, dtype=np.float32) + 0.5) * (src_size / size) - 0.5
    coords = coords - (src_size - 1.0) / 2.0
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    kernel = np.clip(1.0 - np.sqrt(xx ** 2 + yy ** 2) / 50.0, 0.0, 1.0).astype(np.float32)
    _CIRCLE_MASK_CACHE[key] = kernel
    return kernel


def _stamp_grid(canvas: np.ndarray,
                      traj_grid: np.ndarray,
                      amplitude,
                      *,
                      radius: int,
                      kernel: np.ndarray,
                      decay: bool,
                      clip: bool) -> None:
    traj_grid = np.asarray(traj_grid, dtype=np.int32)
    if np.isscalar(amplitude):
        amps = np.full(len(traj_grid), float(amplitude), dtype=np.float32)
    else:
        amps = np.asarray(amplitude, dtype=np.float32)
        if amps.ndim == 0:
            amps = np.full(len(traj_grid), float(amps), dtype=np.float32)
    if decay:
        decay_curve = np.interp(
            np.linspace(0.0, 1.0, len(traj_grid), dtype=np.float32),
            np.array([0.0, 0.7, 1.0], dtype=np.float32),
            np.array([1.0, 1.0, 0.5], dtype=np.float32),
        )
        amps = amps * decay_curve

    h, w = canvas.shape
    for (x, y), amp in zip(traj_grid, amps):
        if clip:
            x0, x1 = max(0, x - radius), min(h, x + radius + 1)
            y0, y1 = max(0, y - radius), min(w, y + radius + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            kx0, kx1 = x0 - (x - radius), kernel.shape[0] - ((x + radius + 1) - x1)
            ky0, ky1 = y0 - (y - radius), kernel.shape[1] - ((y + radius + 1) - y1)
            canvas[x0:x1, y0:y1] += amp * kernel[kx0:kx1, ky0:ky1]
        else:
            if (x - radius < 0 or y - radius < 0 or
                    x + radius + 1 >= h or y + radius + 1 >= w):
                continue
            canvas[x - radius:x + radius + 1, y - radius:y + radius + 1] += amp * kernel


def _stamp_trajectories(canvas: np.ndarray,
                     trajs_grid: np.ndarray,
                     *,
                     amplitudes,
                     radii,
                     decay: bool,
                     clip: bool,
                     mask_root: str | None = None) -> np.ndarray:
    if len(trajs_grid) == 0:
        return canvas
    trajs_grid = np.asarray(trajs_grid, dtype=np.int32)
    if trajs_grid.ndim == 2:
        trajs_grid = trajs_grid[None, :, :]
    amps = np.asarray(amplitudes, dtype=np.float32)
    if amps.ndim == 0:
        amps = np.full(len(trajs_grid), float(amps), dtype=np.float32)
    rads = np.asarray(radii, dtype=np.int32)
    if rads.ndim == 0:
        rads = np.full(len(trajs_grid), int(rads), dtype=np.int32)

    out = canvas.copy()
    for traj, amp, radius in zip(trajs_grid, amps, rads):
        kernel = _circle_mask_kernel(int(radius), mask_root=mask_root)
        _stamp_grid(
            out, traj, amp,
            radius=int(radius), kernel=kernel, decay=decay, clip=clip,
        )
    return out


def _trajectory_cut_one(map_arr: np.ndarray, center: np.ndarray, half_size: int) -> np.ndarray:
    a, b = map_arr.shape
    c = center.astype(np.int32).copy()
    c[0] = min(max(c[0], half_size), a - half_size)
    c[1] = min(max(c[1], half_size), b - half_size)
    return map_arr[c[0] - half_size:c[0] + half_size,
                   c[1] - half_size:c[1] + half_size]


def _fill_missing_trajectory(traj: np.ndarray, init_position: float = 10000.0) -> np.ndarray | None:
    traj = traj.copy()
    if traj.max() < init_position / 2:
        return traj
    valid = np.where(traj[:, 0] <= init_position / 2)[0]
    if len(valid) == 0:
        return None
    traj[:valid[0]] = traj[valid[0]]
    traj[valid[-1]:] = traj[valid[-1]]
    return traj


def prepare_ethucy(test_scene: str = "eth",
                   data_root: str | None = None,
                   obs_len: int = 8, pred_len: int = 12,
                   rotate_time_frame: int = 6,
                   padding: str = "mean",
                   obs_input_dim: int = 6,
                   load_scene_heatmap: bool = False,
                   scene_grid: int = 100):
    """ETH-UCY standard leave-one-scene-out preprocessing."""
    if padding not in {"mean", "zero", "linear"}:
        raise ValueError(f"Unknown DCT padding mode: {padding}")

    def _build(obs, pred):
        last = obs[:, -1:, :]
        obs_rel = obs - last
        pred_rel = pred - last
        obs_rel_r, pred_rel_r, obs_abs_r, rot = _heading_rotate_eth(
            obs_rel, pred_rel, obs, rotate_time_frame,
        )
        if obs_input_dim == 2:
            obs_input = obs_rel_r                                         # [N, T, 2]
        elif obs_input_dim == 6:
            obs_vel = torch.cat([
                obs_rel_r[:, 1:] - obs_rel_r[:, :-1],
                torch.zeros_like(obs_rel_r[:, -1:]),
            ], dim=1)
            obs_input = torch.cat([obs_abs_r, obs_rel_r, obs_vel], dim=-1)   # [N, T, 6]
        else:
            raise ValueError(f"obs_input_dim must be 2 or 6 for ETH-UCY, got {obs_input_dim}")
        scale = torch.ones(len(obs), 1, dtype=obs.dtype, device=obs.device)
        return obs_input, pred_rel_r, rot, last, scale, "relative_last"

    standard_root = _resolve_standard_root(data_root, test_scene)
    train_obs, train_pred, test_obs, test_pred = _standard_protocol_split(
        standard_root, test_scene, obs_len=obs_len, pred_len=pred_len,
    )

    train_input, train_target, train_rot, _, _, _ = _build(train_obs, train_pred)
    test_input, _, test_rot, test_last, test_scale, test_recover_mode = _build(test_obs, test_pred)

    target = dct_1d(train_target)
    target_norm, mn, mx = _minmax_normalise(target)

    data = {
        "train_obs":      train_input.to(device),
        "train_dct_norm": target_norm.to(device),
        "train_nbr_obs":  None,
        "train_nbr_mask": None,
        "test_obs":       test_input.to(device),
        "test_pred":      test_pred.to(device),
        "test_nbr_obs":   None,
        "test_nbr_mask":  None,
        # _heading_rotate_eth uses column-vec rotation (``obs_r = rot @ obs``);
        # to undo we need ``rot.T @ pred_col``, which trainer's row-vec einsum
        # ``pred @ rot_inv`` computes when rot_inv = rot (so rot_inv.T = rot.T).
        "test_rot_inv":   test_rot.to(device) if test_rot is not None else None,
        "test_scale":     test_scale.to(device) if test_scale is not None else None,
        "test_last_obs":  test_last.to(device),
        "test_recover_mode": test_recover_mode,
    }
    if load_scene_heatmap:
        cache_dir = os.path.join(_standard_cache_root(data_root), ".lrsfm_cache", "trajectory_standard")
        data["train_scene_map"] = _standard_context_maps(
            standard_root, test_scene, "train",
            obs_len=obs_len, pred_len=pred_len, grid=scene_grid,
            expected_count=len(train_obs),
            cache_path=os.path.join(
                cache_dir, f"{test_scene}_train_g{scene_grid}_o{obs_len}_p{pred_len}_std.npy",
            ),
        )
        data["test_scene_map"] = _standard_context_maps(
            standard_root, test_scene, "test",
            obs_len=obs_len, pred_len=pred_len, grid=scene_grid,
            expected_count=len(test_obs),
            cache_path=os.path.join(
                cache_dir, f"{test_scene}_test_g{scene_grid}_o{obs_len}_p{pred_len}_std.npy",
            ),
        )

    stats = {
        "minmax_min":     mn.to(device),
        "minmax_max":     mx.to(device),
        "train_dct_mean": target.reshape(-1, *target.shape[-2:]).mean(0).to(device),
        "padding": padding,
        "trajectory_preprocess": "heading_6d",
        "train_dct_full": target.to(device),
    }
    return data, stats


# ----------------------------------------------------------------------
# SDD
# ----------------------------------------------------------------------
def prepare_sdd(data_root: str | None = None,
                padding: str = "mean",
                obs_input_dim: int = 2,
                load_scene_heatmap: bool = False,
                scene_grid: int = 100):
    """SDD preprocessing under the standard SimAug split."""
    if obs_input_dim not in {2, 6}:
        raise ValueError(f"obs_input_dim must be 2 or 6 for SDD, got {obs_input_dim}")
    if padding not in {"mean", "zero", "linear"}:
        raise ValueError(f"Unknown DCT padding mode: {padding}")
    standard_root = _resolve_standard_root(data_root, "sdd")
    train_obs, train_pred, test_obs, test_pred = _standard_protocol_split(
        standard_root, "sdd", obs_len=8, pred_len=12,
    )
    train_nbr_obs = torch.zeros(len(train_obs), 0, 8, 2)
    train_nbr_mask = torch.zeros(len(train_obs), 0, dtype=torch.bool)
    test_nbr_obs = torch.zeros(len(test_obs), 0, 8, 2)
    test_nbr_mask = torch.zeros(len(test_obs), 0, dtype=torch.bool)

    train_last = train_obs[:, -1:, :]
    test_last  = test_obs[:, -1:, :]
    train_obs_rel = train_obs - train_last
    train_pred_rel = train_pred - train_last
    test_obs_rel  = test_obs - test_last
    test_pred_rel = test_pred - test_last
    train_nbr_rel = train_nbr_obs - train_last.unsqueeze(1)
    test_nbr_rel  = test_nbr_obs  - test_last.unsqueeze(1)

    train_obs_rot, train_pred_rot, train_nbr_rot, train_rot = _heading_rotate_sdd(train_obs_rel, train_pred_rel, train_nbr_rel)
    test_obs_rot,  _,             test_nbr_rot, test_rot_inv = _heading_rotate_sdd(test_obs_rel,  test_pred_rel,  test_nbr_rel)

    if obs_input_dim == 2:
        train_input = train_obs_rot
        test_input = test_obs_rot
        train_nbr_input = train_nbr_rot
        test_nbr_input = test_nbr_rot
    else:
        # Context representation: rotated absolute trajectory,
        # last-observation-relative trajectory, and relative velocity.
        train_obs_abs_rot = torch.einsum("bij,btj->bti", train_rot, train_obs)
        test_obs_abs_rot = torch.einsum("bij,btj->bti", test_rot_inv, test_obs)
        train_vel = torch.cat([
            train_obs_rot[:, 1:] - train_obs_rot[:, :-1],
            torch.zeros_like(train_obs_rot[:, -1:]),
        ], dim=1)
        test_vel = torch.cat([
            test_obs_rot[:, 1:] - test_obs_rot[:, :-1],
            torch.zeros_like(test_obs_rot[:, -1:]),
        ], dim=1)
        train_input = torch.cat([train_obs_abs_rot, train_obs_rot, train_vel], dim=-1)
        test_input = torch.cat([test_obs_abs_rot, test_obs_rot, test_vel], dim=-1)

        if train_nbr_obs.shape[1] == 0:
            train_nbr_abs_rot = train_nbr_obs.clone()
            test_nbr_abs_rot = test_nbr_obs.clone()
        else:
            train_nbr_abs_rot = torch.einsum(
                "bij,bmtj->bmti", train_rot, train_nbr_obs,
            )
            test_nbr_abs_rot = torch.einsum(
                "bij,bmtj->bmti", test_rot_inv, test_nbr_obs,
            )
        train_nbr_vel = torch.cat([
            train_nbr_rot[:, :, 1:] - train_nbr_rot[:, :, :-1],
            torch.zeros_like(train_nbr_rot[:, :, -1:]),
        ], dim=2)
        test_nbr_vel = torch.cat([
            test_nbr_rot[:, :, 1:] - test_nbr_rot[:, :, :-1],
            torch.zeros_like(test_nbr_rot[:, :, -1:]),
        ], dim=2)
        train_nbr_input = torch.cat([train_nbr_abs_rot, train_nbr_rot, train_nbr_vel], dim=-1)
        test_nbr_input = torch.cat([test_nbr_abs_rot, test_nbr_rot, test_nbr_vel], dim=-1)

    train_dct = dct_1d(train_pred_rot)
    train_dct_norm, mn, mx = _minmax_normalise(train_dct)

    data = {
        "train_obs":      train_input.to(device),
        "train_dct_norm": train_dct_norm.to(device),
        "train_nbr_obs":  train_nbr_input.to(device),
        "train_nbr_mask": train_nbr_mask.to(device),
        "test_obs":       test_input.to(device),
        "test_pred":      test_pred.to(device),
        "test_nbr_obs":   test_nbr_input.to(device),
        "test_nbr_mask":  test_nbr_mask.to(device),
        "test_rot_inv":   test_rot_inv.to(device),
        "test_last_obs":  test_last.to(device),
    }
    if load_scene_heatmap:
        cache_dir = os.path.join(_standard_cache_root(data_root), ".lrsfm_cache", "trajectory_standard")
        data["train_scene_map"] = _standard_context_maps(
            standard_root, "sdd", "train",
            obs_len=8, pred_len=12, grid=scene_grid,
            expected_count=len(train_obs),
            cache_path=os.path.join(cache_dir, f"sdd_train_g{scene_grid}_o8_p12_std.npy"),
        )
        data["test_scene_map"] = _standard_context_maps(
            standard_root, "sdd", "test",
            obs_len=8, pred_len=12, grid=scene_grid,
            expected_count=len(test_obs),
            cache_path=os.path.join(cache_dir, f"sdd_test_g{scene_grid}_o8_p12_std.npy"),
        )
    stats = {
        "minmax_min": mn.to(device),
        "minmax_max": mx.to(device),
        "train_dct_mean": train_dct.reshape(-1, *train_dct.shape[-2:]).mean(0).to(device),
        "padding": padding,
        "train_dct_full": train_dct.to(device),
    }
    return data, stats


# ----------------------------------------------------------------------
# NBA (A=11 joint prediction)
# ----------------------------------------------------------------------
def prepare_nba(data_root: str = "data/pedestrian/nba", padding: str = "mean"):
    """6-dim encoder input (abs-centered, ego-rel, velocity), no rotation."""
    if padding not in {"mean", "zero", "linear"}:
        raise ValueError(f"Unknown DCT padding mode: {padding}")
    traj_scale = 94.0 / 28.0
    court_center = torch.tensor([14.0, 7.5])
    train_raw = np.load(os.path.join(data_root, "nba_train.npy"))
    test_raw  = np.load(os.path.join(data_root, "nba_test.npy"))

    def _process(raw):
        d = torch.from_numpy(raw).float() / traj_scale  # [N, 30, 11, 2]
        d = d.permute(0, 2, 1, 3)                        # [N, 11, 30, 2]
        obs = d[:, :, :10, :]
        pred = d[:, :, 10:, :]
        last_obs = obs[:, :, -1:, :]                     # [N, 11, 1, 2]

        obs_rel = obs - last_obs
        obs_ctr = obs - court_center
        obs_vel = torch.cat([obs_rel[:, :, 1:] - obs_rel[:, :, :-1],
                             torch.zeros_like(obs_rel[:, :, -1:])], dim=2)
        obs_input = torch.cat([obs_ctr, obs_rel, obs_vel], dim=-1)  # [N, 11, 10, 6]

        # Neighbours: per-agent, the other 10
        nbr_list = [obs[:, [j for j in range(11) if j != i]] for i in range(11)]
        nbr_obs = torch.stack(nbr_list, dim=1)                       # [N, 11, 10, 10, 2]
        nbr_rel = nbr_obs - last_obs.unsqueeze(2)
        nbr_ctr = nbr_obs - court_center
        nbr_vel = torch.cat([nbr_rel[:, :, :, 1:] - nbr_rel[:, :, :, :-1],
                             torch.zeros_like(nbr_rel[:, :, :, -1:])], dim=3)
        nbr_input = torch.cat([nbr_ctr, nbr_rel, nbr_vel], dim=-1)  # [N, 11, 10, 10, 6]
        nbr_mask = torch.ones(d.shape[0], 11, 10, dtype=torch.bool)

        pred_rel = pred - last_obs                                  # absolute → rel
        return obs_input, pred, pred_rel, nbr_input, nbr_mask, last_obs

    train_obs, _, train_pred_rel, train_nbr, train_mask, _ = _process(train_raw)
    test_obs,  test_pred_abs,  _,              test_nbr,  test_mask,  test_last  = _process(test_raw)

    # DCT per-agent
    N_train = train_pred_rel.shape[0]
    train_dct_per_agent = dct_1d(train_pred_rel.reshape(-1, 20, 2)).reshape(N_train, 11, 20, 2)
    train_dct_norm, mn, mx = _minmax_normalise(train_dct_per_agent)

    data = {
        "train_obs":      train_obs.to(device),
        "train_dct_norm": train_dct_norm.to(device),
        "train_nbr_obs":  train_nbr.to(device),
        "train_nbr_mask": train_mask.to(device),
        "test_obs":       test_obs.to(device),
        "test_pred":      test_pred_abs.to(device),
        "test_nbr_obs":   test_nbr.to(device),
        "test_nbr_mask":  test_mask.to(device),
        "test_rot_inv":   None,                          # NBA has no rotation
        "test_last_obs":  test_last.to(device),
    }
    train_dct_all = dct_1d(train_pred_rel.reshape(-1, 20, 2))
    stats = {
        "minmax_min": mn.to(device),
        "minmax_max": mx.to(device),
        "train_dct_mean": train_dct_all.mean(0).to(device),
        "padding": padding,
        "train_dct_full": train_dct_per_agent.to(device),
    }
    return data, stats
