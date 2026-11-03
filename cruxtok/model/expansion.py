import torch
import torch.nn as nn

from cruxtok.model.layers import Attention, window_prior


class Expansion(nn.Module):
    """R_k: L learned position queries read the l_k code vectors."""

    def __init__(self, n_slots, length, d, n_heads, dropout=0.0, attn_prior=True, query_init_std=0.02, prior_sharpness=4.0, qk_norm=False):
        super().__init__()
        self.positions = nn.Parameter(torch.empty(length, d))
        self.slots = nn.Parameter(torch.empty(n_slots, d))
        for p in (self.positions, self.slots):
            nn.init.trunc_normal_(p, std=query_init_std)
        self.prior = nn.Parameter(window_prior(length, n_slots, prior_sharpness)) if attn_prior else None
        self.ln_key = nn.LayerNorm(d)
        self.attn = Attention(d, n_heads, dropout, qk_norm)

    def forward(self, e):
        q = self.positions.expand(e.shape[0], -1, -1)
        return self.attn(q, self.ln_key(e) + self.slots, e, score_bias=self.prior)[0]
