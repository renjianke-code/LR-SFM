import torch
import torch.nn as nn


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation y = x * (1 + scale) + shift, broadcasting over K, A axes."""
    if x.dim() == 4:
        return x * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)
    if x.dim() == 3:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return x * (1 + scale) + shift


class MTRDecoderBlock(nn.Module):
    """K-to-K + A-to-A self-attention stack with AdaLN time modulation."""

    def __init__(self, d_model: int = 128, nhead: int = 4, num_blocks: int = 2,
                 dropout: float = 0.1, norm_first: bool = True,
                 compat_mode: bool = False):
        super().__init__()
        self.num_blocks = num_blocks
        self.compat_mode = compat_mode
        self.self_attn_K = nn.ModuleList()
        self.self_attn_A = nn.ModuleList()
        self.t_adaLN = nn.ModuleList()
        for _ in range(num_blocks):
            self.self_attn_K.append(nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=dropout, norm_first=norm_first, batch_first=True,
            ))
            self.self_attn_A.append(nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=dropout, norm_first=norm_first, batch_first=True,
            ))
            adaln = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
            nn.init.constant_(adaln[-1].weight, 0)
            nn.init.constant_(adaln[-1].bias, 0)
            self.t_adaLN.append(adaln)

    def forward(self, query_token: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """query_token: [B, K, A, D]; time_emb: [B, D] -> [B, K, A, D]."""
        B, K, A = query_token.shape[:3]
        cur = query_token
        for i in range(self.num_blocks):
            shift, scale = self.t_adaLN[i](time_emb).chunk(2, dim=-1)
            cur = _modulate(cur, shift, scale)
            # K-to-K
            source = query_token if self.compat_mode else cur
            cur = source.permute(0, 2, 1, 3).reshape(B * A, K, -1)
            cur = self.self_attn_K[i](cur)
            cur = cur.reshape(B, A, K, -1).permute(0, 2, 1, 3)
            # A-to-A
            cur = cur.reshape(B * K, A, -1)
            cur = self.self_attn_A[i](cur)
            cur = cur.reshape(B, K, A, -1)
        return cur
