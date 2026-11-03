import torch
import torch.nn as nn

from cruxtok.model.layers import Attention, Block, window_prior


class QueryPool(nn.Module):
    """Learned queries attend to the residual; values keep the residual's scale, which shrinks with depth."""

    def __init__(self, n_queries, length, d, n_heads, dropout=0.0, self_attn=False, attn_prior=True, query_init_std=0.02, prior_sharpness=4.0, qk_norm=False):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(n_queries, d))
        self.level = nn.Parameter(torch.empty(1, d))
        for p in (self.queries, self.level):
            nn.init.trunc_normal_(p, std=query_init_std)
        # Plain queries collapse onto one target. The prior starts from NCM's windows and stays learnable.
        self.prior = nn.Parameter(window_prior(n_queries, length, prior_sharpness)) if attn_prior else None
        self.ln_key = nn.LayerNorm(d)
        self.attn = Attention(d, n_heads, dropout, qk_norm)
        self.self_attn = Block(d, n_heads, dropout) if self_attn else None

    def forward(self, residual, key_padding_mask=None, need_weights=False):
        q = (self.queries + self.level).expand(residual.shape[0], -1, -1)
        u, weights = self.attn(q, self.ln_key(residual), residual, key_padding_mask, need_weights=need_weights, score_bias=self.prior)
        if self.self_attn is not None:
            u = self.self_attn(u)
        return u, weights
