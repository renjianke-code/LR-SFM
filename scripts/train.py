"""Train LR-SFM from a YAML config.

Examples:
    python scripts/train.py --config config/ethucy_eth.yaml
    python scripts/train.py --config config/sdd.yaml
    python scripts/train.py --config config/nba.yaml
"""
import argparse
import os
import sys
from dataclasses import replace

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lr_sfm import LRSFM, prepare_ethucy, prepare_sdd, prepare_nba, train, evaluate
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


def _save_path(cfg, ckpt_dir):
    if cfg.dataset == "ethucy":
        return os.path.join(ckpt_dir, "ethucy", f"{cfg.scene}.pt")
    return os.path.join(ckpt_dir, f"{cfg.dataset}.pt")


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


def _std_init_margins(cfg, data, out_modes):
    if cfg.sdl_margin_from_std_scale <= 0:
        return cfg.ctr_init_margins

    train_modes = data["train_dct_norm"]
    if train_modes.dim() == 4:
        train_modes = train_modes[:, :, :out_modes, :]
    else:
        train_modes = train_modes[:, None, :out_modes, :]
    std = train_modes.std(dim=(0, 1, 3), unbiased=False).clamp_min(1e-4)
    return (std * float(cfg.sdl_margin_from_std_scale)).detach().cpu().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=None, help="override cfg.epochs")
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--steps", type=int, default=3,
                        help="Euler steps used for training-time evaluation (best-ckpt selection)")
    parser.add_argument("--eval-every", type=int, default=1,
                        help="Evaluate every N epochs during training")
    parser.add_argument("--solver", choices=["euler", "lin_poly"], default="euler")
    parser.add_argument("--lin-poly-p", type=int, default=5)
    parser.add_argument("--lin-poly-long-step", type=int, default=1000)
    parser.add_argument("--untied-noise", action="store_true",
                        help="use independent initial noise for each K hypothesis during evaluation/selection")
    parser.add_argument("--selection-metric", choices=["ade", "fde", "ade_fde"], default="ade",
                        help="metric used for training-time best checkpoint selection")
    parser.add_argument("--data-root", default="data/pedestrian")
    parser.add_argument("--ckpt-dir", default="ckpt")
    parser.add_argument("--init-ckpt", default=None,
                        help="optional checkpoint to initialize model weights from")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--set", dest="overrides", action="append", default=["padding=linear"],
                        help="override config field, e.g. --set use_scene=false")
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = _apply_cli_overrides(load_config(args.config), args.overrides)
    name = f"{cfg.dataset}-{cfg.scene}" if cfg.scene else cfg.dataset

    data, stats = _prepare(cfg, args.data_root)

    out_modes = cfg.L
    margins = _std_init_margins(cfg, data, out_modes)
    model = LRSFM(
        ctr_init_margins=margins,
        L=out_modes, K=args.K, A=cfg.A,
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
    if args.init_ckpt is not None:
        ckpt = torch.load(args.init_ckpt, map_location=device, weights_only=True)
        state = ckpt.get("ema", ckpt)
        model.load_state_dict(state, strict=True)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    forward_kwargs = {
        "contrastive_weight": cfg.contrastive_weight,
        "loss_reg_reduction": cfg.loss_reg_reduction,
        "sdl_skip_best_m":    cfg.sdl_skip_best_m,
        "sdl_skip_worst_m":   cfg.sdl_skip_worst_m,
        "sdl_std_pull_weight":      cfg.sdl_std_pull_weight,
        "sdl_std_pull_scale":       cfg.sdl_std_pull_scale,
    }
    eval_fn = lambda m: evaluate(
        m, data, stats, K=args.K, steps=args.steps,
        solver=args.solver,
        lin_poly_p=args.lin_poly_p,
        lin_poly_long_step=args.lin_poly_long_step,
        tied_noise=not args.untied_noise,
    )

    best = train(
        model, data, stats,
        epochs=args.epochs or cfg.epochs, bs=cfg.bs, lr=cfg.lr,
        weight_decay=cfg.weight_decay, optimizer=cfg.optimizer,
        ema_decay=cfg.ema_decay,
        forward_kwargs=forward_kwargs,
        eval_fn=eval_fn, eval_every=args.eval_every,
        selection_metric=args.selection_metric,
        warmup_frac=cfg.warmup_frac,
        sdl_stop_epoch=cfg.sdl_stop_epoch,
        save_path=_save_path(cfg, args.ckpt_dir), name=name,
    )

    res = evaluate(
        best, data, stats, K=args.K, steps=args.steps,
        solver=args.solver,
        lin_poly_p=args.lin_poly_p,
        lin_poly_long_step=args.lin_poly_long_step,
        tied_noise=not args.untied_noise,
    )
    print(f"  >> {name}: minADE={res['minADE']:.4f}  minFDE={res['minFDE']:.4f}")


if __name__ == "__main__":
    main()
