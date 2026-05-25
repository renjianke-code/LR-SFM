"""LR-SFM: Low-Rank Spectral Flow Matching for Human Trajectory Prediction.

The model consists of a Social Transformer encoder, an MTR-style K/A self-attention
decoder, multi-query flow matching with WTA selection, and Spectral Diversity Loss
with per-mode learnable margins.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import (
    SocialObsEncoder,
    TrajectoryObsEncoder,
)
from .decoder import MTRDecoderBlock


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for the FM time step."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -freq)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class LRSFM(nn.Module):
    """Low-Rank Spectral Flow Matching with Spectral Diversity Loss.

    Operates on truncated DCT coefficients (first ``L`` modes) and trains
    ``K`` queries jointly with Winner-Takes-All selection and per-mode
    learnable contrastive margins.
    """

    def __init__(
        self,
        ctr_init_margins,                       # list[float], length = L
        L: int = 4,
        K: int = 20,
        A: int = 1,
        d_model: int = 128,
        decoder_blocks: int = 2,
        decoder_nhead: int = 4,
        decoder_dropout: float = 0.1,
        enc_nhead: int = 4,
        enc_layers: int = 4,
        social_nhead: int = 4,
        social_layers: int = 2,
        logit_norm_mean: float = -0.5,
        logit_norm_std: float = 1.5,
        drop_logi_k: float = 20.0,
        drop_logi_m: float = 0.5,
        cls_weight: float = 1.0,
        obs_len: int = 8,
        obs_input_dim: int = 2,
        encoder_type: str = "social",
        use_scene: bool = False,
        scene_grid: int = 100,
        compat_decoder: bool = False,
    ):
        super().__init__()
        if L < 1:
            raise ValueError("L must be at least 1")
        if K < 1:
            raise ValueError("K must be at least 1")
        if A < 1:
            raise ValueError("A must be at least 1")
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for sinusoidal time embeddings")

        self.out_dim = L * 2
        self.L = L
        self.K = K
        self.A = A
        self.d_model = d_model
        self.logit_norm_mean = logit_norm_mean
        self.logit_norm_std = logit_norm_std
        self.drop_logi_k = drop_logi_k
        self.drop_logi_m = drop_logi_m
        self.cls_weight = cls_weight
        self.encoder_type = encoder_type
        self.compat_decoder = compat_decoder

        # Encoder
        if encoder_type == "trajectory":
            self.enc = TrajectoryObsEncoder(
                obs_len=obs_len, input_dim=obs_input_dim, d_model=d_model,
                nhead=enc_nhead, num_layers=enc_layers, max_agents=A,
                use_scene_map=use_scene, scene_grid=scene_grid,
            )
        elif encoder_type == "social":
            self.enc = SocialObsEncoder(
                obs_len=obs_len, input_dim=obs_input_dim, d_model=d_model,
                enc_nhead=enc_nhead, enc_layers=enc_layers,
                social_nhead=social_nhead, social_layers=social_layers,
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        self.use_scene = use_scene

        # Query and agent positional embeddings
        self.motion_query_emb = nn.Embedding(K, d_model)
        self.agent_order_emb = nn.Embedding(A, d_model)

        # FM time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Noisy DCT-coefficient embedding
        self.noisy_y_mlp = nn.Sequential(
            nn.Linear(self.out_dim, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Pre-decoder self-attention on K and A axes
        noisy_nhead = 4 if compat_decoder else decoder_nhead
        noisy_norm_first = False if compat_decoder else True
        self.noisy_y_attn_k = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=noisy_nhead, dim_feedforward=d_model * 4,
            dropout=decoder_dropout, norm_first=noisy_norm_first, batch_first=True,
        )
        self.noisy_y_attn_a = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=noisy_nhead, dim_feedforward=d_model * 4,
            dropout=decoder_dropout, norm_first=noisy_norm_first, batch_first=True,
        )

        # Fuse [ctx, y_emb, t_emb] -> d_model
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.LayerNorm(d_model), nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.post_pe_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # MTR decoder (K/A self-attention with AdaLN time modulation)
        self.decoder = MTRDecoderBlock(
            d_model, decoder_nhead, decoder_blocks, decoder_dropout,
            norm_first=not compat_decoder,
            compat_mode=compat_decoder,
        )

        # Heads
        if compat_decoder:
            self.reg_head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model * 2), nn.ReLU(),
                nn.Linear(d_model * 2, self.out_dim),
            )
            self.cls_head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, 1),
            )
        else:
            self.reg_head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, self.out_dim),
            )
            self.cls_head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, 1),
            )

        # Per-mode learnable margins (Spectral Diversity Loss)
        if len(ctr_init_margins) != L:
            raise ValueError("ctr_init_margins must match L (DCT truncation order)")
        init_margins = torch.tensor(
            [max(float(m), 1e-4) for m in ctr_init_margins],
            dtype=torch.float32,
        )
        init_vals = []
        for m in init_margins.tolist():
            init_vals.append(math.log(math.exp(m) - 1.0) if m > 0.01 else -10.0)
        init_vals = torch.tensor(init_vals)
        self.mode_margin_logits = nn.Parameter(
            init_vals,
            requires_grad=True,
        )

    def _sdl_margins(self) -> torch.Tensor:
        return F.softplus(self.mode_margin_logits)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _encode(self, obs, neighbor_obs, neighbor_mask, scene_heatmap=None):
        """Return ctx [B,D] for A=1, or [B,A,D] for A>1. Optionally fuses scene_ctx."""
        if self.A > 1 and obs.dim() == 4:
            B = obs.shape[0]
            obs_flat = obs.reshape(B * self.A, *obs.shape[2:])
            nbr_flat = neighbor_obs.reshape(B * self.A, *neighbor_obs.shape[2:]) if neighbor_obs is not None else None
            mask_flat = neighbor_mask.reshape(B * self.A, *neighbor_mask.shape[2:]) if neighbor_mask is not None else None
            ego_ctx = self.enc(obs_flat, nbr_flat, mask_flat).reshape(B, self.A, -1)
        else:
            ego_ctx = self.enc(obs, neighbor_obs, neighbor_mask, scene_heatmap=scene_heatmap)
        return ego_ctx

    def _run_network(self, y_t, t, ctx, training: bool):
        """y_t: [B,K,A,out_dim], t: [B], ctx: [B,D] or [B,A,D] -> pred_x1, pred_cls."""
        B, K, A = y_t.shape[:3]
        dev = y_t.device

        y_emb = self.noisy_y_mlp(y_t)

        t_emb = self.time_mlp(t * 1000.0)
        t_batch = t_emb[:, None, None, :].expand(B, K, A, -1)

        k_pe = self.motion_query_emb(torch.arange(K, device=dev))[None, :, None, :].expand(B, K, A, -1)
        a_pe = self.agent_order_emb(torch.arange(A, device=dev))[None, None, :, :].expand(B, K, A, -1)

        # K/A self-attention on noisy embeddings
        y_emb = y_emb + k_pe + a_pe
        y_emb = self.noisy_y_attn_k(
            y_emb.permute(0, 2, 1, 3).reshape(B * A, K, -1)
        ).reshape(B, A, K, -1).permute(0, 2, 1, 3)
        y_emb = self.noisy_y_attn_a(
            y_emb.reshape(B * K, A, -1)
        ).reshape(B, K, A, -1)

        # Time-dependent embedding dropout (training only)
        if training:
            p_m = torch.sigmoid(self.drop_logi_k * (t - self.drop_logi_m))
            y_emb = y_emb.masked_fill(
                torch.rand(B, 1, 1, 1, device=dev) < p_m[:, None, None, None], 0.0,
            )

        if ctx.dim() == 2:
            ctx_batch = ctx[:, None, None, :].expand(B, K, A, -1)
        else:
            ctx_batch = ctx[:, None, :, :].expand(B, K, A, -1)
        fused = self.fusion_mlp(torch.cat([ctx_batch, y_emb, t_batch], dim=-1))
        query = self.post_pe_mlp(fused + k_pe + a_pe)
        readout = self.decoder(query, t_emb)

        return self.reg_head(readout), self.cls_head(readout).squeeze(-1)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(
        self,
        obs,
        x1,
        neighbor_obs=None,
        neighbor_mask=None,
        dct_min=None,
        dct_max=None,
        contrastive_weight: float = 0.5,
        loss_reg_reduction: str = "mean",
        sdl_skip_best_m: int = 0,
        sdl_skip_worst_m: int = 0,
        sdl_std_pull_weight: float = 0.0,
        sdl_std_pull_scale: float = 1.0,
        sdl_mode_std=None,
        scene_heatmap=None,
    ):
        """Training step. Returns (total_loss, reg_loss_val, cls_loss_val).

        Paper loss: L_FM + cls_weight * L_cls + contrastive_weight * L_SDL.
        """
        B = obs.shape[0]
        K, A, L = self.K, self.A, self.L
        dev = obs.device

        # Encode observed trajectory and optional context map.
        ctx = self._encode(obs, neighbor_obs, neighbor_mask, scene_heatmap=scene_heatmap)

        # Sample t ~ sigmoid(Normal(mu, sigma))
        t = torch.sigmoid(torch.randn(B, device=dev) * self.logit_norm_std + self.logit_norm_mean)

        # Build interpolant x_t (tied noise across K queries)
        if A > 1 and x1.dim() == 3:
            x1_exp = x1[:, None, :, :].expand(B, K, A, -1)
        else:
            x1_exp = x1[:, None, None, :].expand(B, K, A, -1)
        noise = torch.randn(B, 1, A, self.out_dim, device=dev).expand(B, K, A, -1)
        t_exp = t[:, None, None, None]
        y_t = (1 - t_exp) * noise + t_exp * x1_exp

        pred_x1, pred_cls = self._run_network(y_t, t, ctx, training=self.training)

        # ------- Regression / classifier loss (in denormalised DCT space) -------
        if dct_min is None or dct_max is None:
            raise ValueError("model.forward() requires dct_min and dct_max for denormalised-DCT WTA loss")
        pred_modes = pred_x1.reshape(B, K, A, L, 2)
        x1_modes = x1_exp.reshape(B, K, A, L, 2)
        mn = dct_min[:L]
        mx = dct_max[:L]
        pred_denorm = (pred_modes + 1.0) / 2.0 * (mx - mn) + mn
        x1_denorm = (x1_modes + 1.0) / 2.0 * (mx - mn) + mn
        err = (pred_denorm - x1_denorm).norm(dim=-1)            # [B,K,A,L]
        if loss_reg_reduction == "mean":
            err_scalar = err.mean(dim=-1)             # [B,K,A]
        elif loss_reg_reduction == "sum":
            err_scalar = err.sum(dim=-1)
        else:
            raise ValueError(f"Unknown loss_reg_reduction: {loss_reg_reduction}")
        err_for_loss = err_scalar
        winner_metric = err_scalar

        winner = winner_metric.argmin(dim=1)                 # [B, A]
        winner_loss = err_for_loss.gather(1, winner.unsqueeze(1).expand(B, 1, A)).squeeze(1)
        loss_reg = winner_loss.mean()

        cls_logits = pred_cls.permute(0, 2, 1).reshape(B * A, K)
        loss_cls = F.cross_entropy(cls_logits, winner.reshape(B * A))

        # ------- Spectral Diversity Loss (winner-relative push) ----------------
        if K > 1:
            margins = self._sdl_margins()                             # [L]
            pred_pm = pred_x1.reshape(B, K, A, L, 2)
            mode_weights = pred_x1.new_ones(L) / L

            winner_gather = winner[:, None, :, None, None].expand(B, 1, A, L, 2)
            center = pred_pm.gather(1, winner_gather).expand_as(pred_pm)
            dist_to_center = (pred_pm - center).norm(dim=-1)         # [B,K,A,L]

            hinge_per_mode = torch.clamp(margins - dist_to_center, min=0)
            hinge_scalar = (hinge_per_mode * mode_weights).sum(dim=-1)  # [B,K,A]

            non_winner_mask = torch.ones(B, A, K, device=dev, dtype=torch.bool)
            non_winner_mask.scatter_(2, winner.unsqueeze(-1), False)
            if sdl_skip_best_m < 0 or sdl_skip_worst_m < 0:
                raise ValueError("sdl_skip_best_m and sdl_skip_worst_m must be non-negative")
            active_non_winners = K - 1
            if sdl_skip_best_m > 0 or sdl_skip_worst_m > 0:
                best_total = min(int(sdl_skip_best_m), K)
                worst_total = min(int(sdl_skip_worst_m), K - best_total)
                skipped_non_winners = max(best_total - 1, 0) + worst_total
                active_non_winners = K - 1 - skipped_non_winners
                if active_non_winners <= 0:
                    raise ValueError("SDL skip mask leaves no active non-winner hypotheses")

                metric_bak = winner_metric.permute(0, 2, 1).detach()
                ranked = metric_bak.argsort(dim=-1, descending=False)
                if best_total > 0:
                    best_idx = ranked[:, :, :best_total]
                    non_winner_mask.scatter_(2, best_idx, False)
                if worst_total > 0:
                    worst_idx = ranked[:, :, K - worst_total:]
                    non_winner_mask.scatter_(2, worst_idx, False)
            loss_ctr = hinge_scalar.permute(0, 2, 1)[non_winner_mask].mean()
            dist_bakl = dist_to_center.permute(0, 2, 1, 3)            # [B,A,K,L]
            nw_dist_mode = dist_bakl[non_winner_mask].reshape(B, A, active_non_winners, L)

            if sdl_std_pull_weight > 0:
                if sdl_std_pull_scale <= 0:
                    raise ValueError("sdl_std_pull_scale must be positive")
                if sdl_mode_std is None:
                    raise ValueError("sdl_std_pull_weight > 0 requires sdl_mode_std")
                std_upper = torch.as_tensor(
                    sdl_mode_std, device=dev, dtype=nw_dist_mode.dtype,
                ).reshape(-1)
                if std_upper.numel() != L:
                    raise ValueError("sdl_mode_std must match L")
                std_upper = std_upper.clamp_min(1e-6) * float(sdl_std_pull_scale)
                loss_std_pull = (
                    F.relu(nw_dist_mode - std_upper.view(1, 1, 1, L)).square()
                    * mode_weights.view(1, 1, 1, L)
                ).sum(dim=-1).mean()
            else:
                loss_std_pull = pred_x1.new_zeros(())
        else:
            zero = pred_x1.new_zeros(())
            loss_ctr = zero
            loss_std_pull = zero

        total = (
            loss_reg
            + self.cls_weight * loss_cls
            + contrastive_weight * loss_ctr
            + sdl_std_pull_weight * loss_std_pull
        )
        return total, loss_reg.item(), loss_cls.item()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    @staticmethod
    def _sampling_schedule(
        steps: int,
        solver: str,
        lin_poly_p: int,
        lin_poly_long_step: int,
    ) -> list[tuple[float, float]]:
        if steps < 1:
            raise ValueError("steps must be at least 1")
        if solver == "euler":
            dt = 1.0 / steps
            return [(i * dt, dt) for i in range(steps)]
        if solver != "lin_poly":
            raise ValueError(f"Unknown sampling solver: {solver}")
        if lin_poly_p < 1:
            raise ValueError("lin_poly_p must be at least 1")
        if lin_poly_long_step < 1:
            raise ValueError("lin_poly_long_step must be at least 1")

        # Two-stage schedule: small linear steps near t=0, then polynomially
        # increasing step sizes.
        n_steps_lin = steps // 2
        n_steps_poly = steps - n_steps_lin
        dt_lin = 1.0 / lin_poly_long_step

        t_lin = [dt_lin * i for i in range(n_steps_lin)]
        dt_lin_ls = [dt_lin] * n_steps_lin

        if n_steps_poly == 0:
            return list(zip(t_lin, dt_lin_ls))

        t_poly_start = (t_lin[-1] + dt_lin) if t_lin else 0.0
        if t_poly_start >= 1.0:
            raise ValueError(
                "lin_poly_long_step is too small for the requested number of steps"
            )
        denom = float(n_steps_poly ** lin_poly_p)
        t_poly = [
            t_poly_start
            + (1.0 - t_poly_start) * float(i ** lin_poly_p) / denom
            for i in range(n_steps_poly + 1)
        ]
        dt_poly = [t_poly[i + 1] - t_poly[i] for i in range(n_steps_poly)]
        return list(zip(t_lin + t_poly[:-1], dt_lin_ls + dt_poly))

    @torch.no_grad()
    def sample(self, obs, K: int = 20, steps: int = 3, neighbor_obs=None, neighbor_mask=None,
               scene_heatmap=None, solver: str = "euler", lin_poly_p: int = 5,
               lin_poly_long_step: int = 1000, tied_noise: bool = True):
        """K-parallel ODE sampling with tied noise.

        Returns:
            x:         [B, K, out_dim] (A=1) or [B, K, A, out_dim] (A>1)
            cls_logits same shape minus last dim
        """
        B = obs.shape[0]
        A = self.A
        dev = obs.device
        if K < 1:
            raise ValueError("K must be at least 1")
        if K > self.K:
            raise ValueError(f"K={K} exceeds the model capacity K={self.K}")

        ctx = self._encode(obs, neighbor_obs, neighbor_mask, scene_heatmap=scene_heatmap)
        if tied_noise:
            x = torch.randn(B, 1, A, self.out_dim, device=dev).expand(B, K, A, -1).clone()
        else:
            x = torch.randn(B, K, A, self.out_dim, device=dev)

        pred_cls = None
        for t_val, dt in self._sampling_schedule(
            steps, solver, lin_poly_p, lin_poly_long_step,
        ):
            t = torch.full((B,), t_val, device=dev)
            pred_x1, pred_cls = self._run_network(x, t, ctx, training=False)
            denom = max(1.0 - t_val, 1e-5)
            x = x + (pred_x1 - x) / denom * dt

        if A > 1:
            return x, pred_cls
        return x.squeeze(2), pred_cls.squeeze(2)
