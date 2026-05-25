"""Evaluate a trained LR-SFM checkpoint with a YAML config.

Examples:
    # Standard best-of-K minADE / minFDE (default K=20, 3 Euler steps)
    python scripts/evaluate.py --config config/ethucy_eth.yaml --ckpt ckpt/ethucy/eth.pt

    # Override sampling steps
    python scripts/evaluate.py --config config/sdd.yaml --ckpt ckpt/sdd.pt --steps 5

    # Diversity metrics (KDE-NLL)
    python scripts/evaluate.py --config config/nba.yaml --ckpt ckpt/nba.pt --kde-nll
"""
import argparse
import os
import sys
from dataclasses import replace

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lr_sfm import LRSFM, prepare_ethucy, prepare_sdd, prepare_nba, evaluate
from lr_sfm.configs import load_config
from lr_sfm.dct import device, set_seed


def _prepare(cfg, data_root):
    if cfg.dataset == "ethucy":
        return prepare_ethucy(test_scene=cfg.scene, data_root=data_root,
                              padding=cfg.padding,
                              obs_input_dim=cfg.obs_input_dim,
                              load_scene_heatmap=cfg.use_scene,
                              scene_grid=cfg.scene_grid)
    if cfg.dataset == "sdd":
        return prepare_sdd(data_root=data_root,
                           padding=cfg.padding,
                           obs_input_dim=cfg.obs_input_dim,
                           load_scene_heatmap=cfg.use_scene,
                           scene_grid=cfg.scene_grid)
    if cfg.dataset == "nba":
        return prepare_nba(data_root=os.path.join(data_root, "nba"), padding=cfg.padding)
    raise ValueError(f"Unknown dataset: {cfg.dataset}")


def _load_checkpoint_state(path):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    return ckpt.get("ema", ckpt)


def _infer_checkpoint_k(state, fallback):
    query_weight = state.get("motion_query_emb.weight")
    if query_weight is None:
        return fallback
    return int(query_weight.shape[0])


def _apply_cli_overrides(cfg, overrides):
    if not overrides:
        return cfg
    data = cfg.__dict__.copy()
    valid = set(data)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got: {item}")
        key, raw = item.split("=", 1)
        if key not in valid:
            raise ValueError(f"Unknown config field in --set: {key}")
        data[key] = yaml.safe_load(raw)
    return replace(cfg, **data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--solver", choices=["euler", "lin_poly"], default="euler")
    parser.add_argument("--lin-poly-p", type=int, default=5)
    parser.add_argument("--lin-poly-long-step", type=int, default=1000)
    parser.add_argument("--untied-noise", action="store_true",
                        help="sample each of the K hypotheses from independent initial noise")
    parser.add_argument("--kde-nll", action="store_true",
                        help="also compute KDE-NLL over K samples")
    parser.add_argument("--data-root", default="data/pedestrian")
    parser.add_argument("--set", dest="overrides", action="append", default=["padding=linear"],
                        help="override config field, e.g. --set use_scene=false")
    args = parser.parse_args()

    set_seed(42)
    cfg = _apply_cli_overrides(load_config(args.config), args.overrides)
    name = f"{cfg.dataset}-{cfg.scene}" if cfg.scene else cfg.dataset

    state = _load_checkpoint_state(args.ckpt)
    model_K = _infer_checkpoint_k(state, args.K)
    if args.K > model_K:
        raise ValueError(f"--K {args.K} exceeds checkpoint capacity K={model_K}")

    data, stats = _prepare(cfg, args.data_root)

    out_modes = cfg.L
    margins = cfg.ctr_init_margins
    model = LRSFM(
        ctr_init_margins=margins,
        L=out_modes, K=model_K, A=cfg.A,
        obs_len=cfg.obs_len, obs_input_dim=cfg.obs_input_dim,
        d_model=cfg.d_model,
        enc_nhead=cfg.enc_nhead, enc_layers=cfg.enc_layers,
        social_nhead=cfg.social_nhead, social_layers=cfg.social_layers,
        decoder_blocks=cfg.decoder_blocks, decoder_nhead=cfg.decoder_nhead,
        encoder_type=cfg.encoder_type,
        cls_weight=cfg.cls_weight,
        use_scene=cfg.use_scene,
        scene_grid=cfg.scene_grid,
        compat_decoder=(cfg.encoder_type == "trajectory"),
    ).to(device)

    model.load_state_dict(state, strict=True)

    res = evaluate(
        model, data, stats, K=args.K, steps=args.steps,
        solver=args.solver,
        lin_poly_p=args.lin_poly_p,
        lin_poly_long_step=args.lin_poly_long_step,
        tied_noise=not args.untied_noise,
        compute_kde_nll=args.kde_nll,
    )
    bar = "─" * 44
    print(f"\n  ╭{bar}╮")
    print(f"  │  {name:18s}    K={args.K:<3d}   steps={args.steps:<3d}   │")
    noise = "untied" if args.untied_noise else "tied"
    print(f"  │  solver={args.solver:12s} noise={noise:6s}          │")
    print(f"  ├{bar}┤")
    print(f"  │      minADE  {res['minADE']:>7.4f}    minFDE  {res['minFDE']:>7.4f}    │")
    print(f"  │      APD     {res['APD']:>7.4f}    CRPS    {res['CRPS']:>7.4f}    │")
    if args.kde_nll:
        print(f"  │      KDE-NLL {res['KDE_NLL']:>7.4f}    meanADE {res['meanADE']:>7.4f}    │")
    print(f"  ╰{bar}╯\n")


if __name__ == "__main__":
    main()
