"""Load per-dataset hyperparameters from YAML config files.

Usage:
    from lr_sfm.configs import load_config
    cfg = load_config("config/ethucy_eth.yaml")
    print(cfg.L, cfg.ctr_init_margins, cfg.lr)
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass
class Config:
    # data + model
    dataset: str
    L: int
    A: int
    obs_len: int
    obs_input_dim: int
    ctr_init_margins: List[float]
    # encoder / decoder shape
    d_model: int
    enc_nhead: int
    enc_layers: int
    social_nhead: int
    social_layers: int
    decoder_blocks: int
    decoder_nhead: int
    # training
    epochs: int
    bs: int
    lr: float
    weight_decay: float
    # SDL loss weight
    contrastive_weight: float
    cls_weight: float = 1.0
    optimizer: str = "adamw"
    ema_decay: float = 0.995
    warmup_frac: float = 0.05
    # If positive, initialize SDL margins from train-split DCT std:
    # margin_l = scale * std_l. This is train-only and avoids a fixed margin
    # being too aggressive for low-variance high-frequency modes.
    sdl_margin_from_std_scale: float = 0.0
    # ETH-UCY only
    scene: str = ""
    # Encoder family. ETH-UCY/SDD configs use "trajectory"; NBA keeps "social".
    encoder_type: str = "social"
    # High-frequency DCT padding used when reconstructing trajectories.
    # Defaults to the mean-padding setting described in the paper.
    padding: str = "mean"
    # Regression reduction over target timesteps/modes.
    loss_reg_reduction: str = "mean"
    # Optional middle-band SDL mask. ``sdl_skip_best_m`` is counted over the
    # full K ranking and therefore includes the winner when positive. This can
    # preserve the best hypotheses while using the middle hypotheses for
    # diversity. ``sdl_skip_worst_m`` leaves far outliers untouched.
    sdl_skip_best_m: int = 0
    sdl_skip_worst_m: int = 0
    # If positive, disable SDL-family losses after this epoch. For example,
    # 40 keeps SDL active through epoch 40 and turns it off from epoch 41.
    sdl_stop_epoch: int = 0
    # Train-statistics pull-back for DCT SDL. If enabled, non-winner
    # hypothesis distances are softly capped by scale * train_std_l in the
    # same space used by SDL distances. This keeps diversity near the empirical
    # DCT support instead of rewarding arbitrary far-away modes.
    sdl_std_pull_weight: float = 0.0
    sdl_std_pull_scale: float = 1.0
    # When True, build per-sample local trajectory context maps from observed
    # ego/neighbor paths and fuse them into the ego context via a small CNN.
    # This is trajectory-derived context, not RGB/semantic HD-map input.
    use_scene: bool = False
    # Grid resolution of the local context map.
    scene_grid: int = 100


_FLOAT_FIELDS = (
    "lr", "weight_decay", "ema_decay", "warmup_frac",
    "cls_weight", "contrastive_weight",
    "sdl_std_pull_weight", "sdl_std_pull_scale",
    "sdl_margin_from_std_scale",
)


def load_config(path: str | Path) -> Config:
    """Read a YAML file and return a populated :class:`Config`.

    PyYAML 1.1 only parses scientific notation as ``float`` when a decimal
    point is present (``1.0e-3`` works, ``1e-3`` becomes a string). We coerce
    numeric fields explicitly so either form is accepted.
    """
    data = yaml.safe_load(Path(path).read_text())
    data.setdefault("social_nhead", 4)
    data.setdefault("social_layers", 2)
    for key in _FLOAT_FIELDS:
        if key in data and isinstance(data[key], str):
            data[key] = float(data[key])
    return Config(**data)
