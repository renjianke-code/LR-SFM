import math

import torch
import torch.nn as nn


class _SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x.float()[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TransformerObsEncoder(nn.Module):
    """Encode observed trajectory [B, T_obs, 2] -> [B, d_model] via Transformer."""

    def __init__(self, obs_len=8, input_dim=2, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, obs_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False,
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, obs):
        x = self.input_proj(obs) + self.pos_emb
        x = self.transformer(x)
        return self.out_norm(x.mean(dim=1))


class SocialTransformer(nn.Module):
    """Ego cross-attends to padded neighbour trajectories."""

    def __init__(self, obs_len=8, input_dim=2, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.ego_proj = nn.Linear(obs_len * input_dim, d_model)
        self.nbr_proj = nn.Linear(obs_len * input_dim, d_model)
        self.ego_norm = nn.LayerNorm(d_model)
        self.nbr_norm = nn.LayerNorm(d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.cross_attn = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, ego_obs, neighbor_obs, neighbor_mask):
        B, A = neighbor_obs.shape[:2]
        ego_emb = self.ego_norm(self.ego_proj(ego_obs.reshape(B, -1))).unsqueeze(1)
        nbr_emb = self.nbr_norm(self.nbr_proj(neighbor_obs.reshape(B, A, -1)))
        all_padded = (~neighbor_mask).all(dim=1)
        social = self.cross_attn(
            ego_emb, nbr_emb, memory_key_padding_mask=~neighbor_mask,
        ).squeeze(1)
        social = self.out_norm(social)
        return social * (~all_padded).float().unsqueeze(1)


class TrajectoryObsEncoder(nn.Module):
    """Trajectory encoder for heading-aligned 6D observation features.

    The 8 observed features are flattened into one token per agent, passed
    through a compact social Transformer, enriched with agent/query positional
    embeddings, and encoded by a Transformer stack.
    """

    def __init__(self, obs_len: int = 8, input_dim: int = 6,
                 hidden_dim: int = 256, d_model: int = 128,
                 nhead: int = 8, num_layers: int = 4, dropout: float = 0.1,
                 social_nhead: int = 2, social_layers: int = 2,
                 max_agents: int = 1, use_scene_map: bool = False,
                 scene_grid: int = 100):
        super().__init__()
        self.use_scene_map = use_scene_map
        self.encode_past = nn.Linear(obs_len * input_dim, hidden_dim, bias=False)
        social_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=social_nhead, dim_feedforward=hidden_dim,
            dropout=dropout, batch_first=True, norm_first=False,
        )
        self.social_transformer = nn.TransformerEncoder(
            social_layer, num_layers=social_layers, enable_nested_tensor=False,
        )
        self.mlp_out = nn.Linear(hidden_dim, d_model)

        self.pos_encoding = nn.Sequential(
            _SinusoidalPosEmb(d_model, theta=10000),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.agent_query_embedding = nn.Embedding(max_agents, d_model)
        self.mlp_pe = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False,
        )
        if self.use_scene_map:
            self.scene_map_encoder = TrajectoryContextMapEncoder(
                obs_len=obs_len, d_model=d_model, scene_grid=scene_grid,
            )

    def forward(self, obs, neighbor_obs=None, neighbor_mask=None, scene_heatmap=None):  # noqa: ARG002
        """``obs``: [B, T, D]  -> ego ctx [B, d_model]."""
        B, T, D = obs.shape
        h = self.encode_past(obs.reshape(B, 1, T * D))     # [B, 1, hidden]
        h = h + self.social_transformer(h)
        h = self.mlp_out(h)                                # [B, 1, d_model]
        if self.use_scene_map and scene_heatmap is not None:
            scene_feature = self.scene_map_encoder(scene_heatmap)
            h = h + scene_feature[:, None, :]

        agent_count = h.shape[1]
        agent_idx = torch.arange(agent_count, device=obs.device)
        pos = self.pos_encoding(agent_idx)
        query = self.agent_query_embedding(agent_idx)
        pe = self.mlp_pe(torch.cat([query, pos], dim=-1))
        h = h + pe.unsqueeze(0)
        return self.transformer_encoder(h).squeeze(1)      # [B, d_model]


class TrajectoryContextMapEncoder(nn.Module):
    """Context-map encoder for LR-SFM trajectory-context fusion.

    Network core:
    ``MaxPool2d(5) -> Flatten -> Linear(obs_len * 64) -> Tanh -> Reshape``.
    LR-SFM then pools the resulting context tokens to one ``d_model`` vector so
    it can be added to the trajectory encoder context.
    """

    def __init__(self, obs_len: int = 8, d_model: int = 128,
                 scene_grid: int = 100, units: int = 64):
        super().__init__()
        self.obs_len = obs_len
        self.units = units
        self.pool = nn.MaxPool2d(kernel_size=5, stride=5)
        pooled = scene_grid // 5
        self.fc = nn.Sequential(
            nn.Linear(pooled * pooled, obs_len * units),
            nn.Tanh(),
        )
        self.token_proj = nn.Identity() if units == d_model else nn.Linear(units, d_model)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, heatmap):
        if heatmap.dtype == torch.uint8:
            heatmap = heatmap.float().div(255.0)
        else:
            heatmap = heatmap.float()
            if heatmap.numel() > 0 and heatmap.detach().max() > 1.0:
                heatmap = heatmap.div(255.0)
        squeeze_out = heatmap.dim() == 2
        if heatmap.dim() == 2:
            heatmap = heatmap.unsqueeze(0).unsqueeze(0)
        elif heatmap.dim() == 3:
            heatmap = heatmap.unsqueeze(1)
        pooled = self.pool(heatmap).flatten(1)
        tokens = self.fc(pooled).reshape(heatmap.shape[0], self.obs_len, self.units)
        ctx = self.token_proj(tokens).mean(dim=1)
        out = self.out_norm(ctx)
        return out.squeeze(0) if squeeze_out else out


class SocialObsEncoder(nn.Module):
    """Ego Transformer + Social cross-attention, fused by a learned gate."""

    def __init__(self, obs_len=8, input_dim=2, d_model=128,
                 enc_nhead=4, enc_layers=2, social_nhead=4, social_layers=2, dropout=0.1):
        super().__init__()
        self.ego_enc = TransformerObsEncoder(
            obs_len=obs_len, input_dim=input_dim, d_model=d_model,
            nhead=enc_nhead, num_layers=enc_layers, dropout=dropout,
        )
        self.social_enc = SocialTransformer(
            obs_len=obs_len, input_dim=input_dim, d_model=d_model,
            nhead=social_nhead, num_layers=social_layers, dropout=dropout,
        )
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.out_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, obs, neighbor_obs=None, neighbor_mask=None):
        ego = self.ego_enc(obs)
        if neighbor_obs is None or neighbor_mask is None:
            return ego
        social = self.social_enc(obs, neighbor_obs, neighbor_mask)
        combined = torch.cat([ego, social], dim=-1)
        g = self.gate(combined)
        return self.out_proj(torch.cat([ego, g * social], dim=-1))
