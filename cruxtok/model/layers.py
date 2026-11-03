import torch
import torch.nn as nn
import torch.nn.functional as F


def padding_positions(ids, pad_id, eos_id):
    """True where a window holds no text. NCM fills windows with EOS, so that is everything after the first EOS."""
    is_eos = ids == eos_id
    return (ids == pad_id) | ((is_eos.long().cumsum(1) - is_eos.long()) > 0)


def window_prior(n_rows, n_cols, sharpness=4.0):
    """Log-attention prior: each of n_rows attends around its evenly spaced share of n_cols, as a fixed window would."""
    rows = (torch.arange(n_rows) + 0.5)[:, None] / n_rows
    cols = (torch.arange(n_cols) + 0.5)[None, :] / n_cols
    width = 0.5 / min(n_rows, n_cols)
    return -sharpness * ((rows - cols) / width).pow(2) / 2


class Attention(nn.Module):
    def __init__(self, d, n_heads, dropout=0.0, qk_norm=False):
        super().__init__()
        assert d % n_heads == 0, f"d={d} must be divisible by n_heads={n_heads}"
        self.n_heads, self.head_dim, self.dropout, self.qk_norm = n_heads, d // n_heads, dropout, qk_norm
        self.q, self.k, self.v, self.out = (nn.Linear(d, d) for _ in range(4))

    def _split(self, x):
        return x.view(x.shape[0], x.shape[1], self.n_heads, self.head_dim).transpose(1, 2)

    # key_padding_mask is True at positions to ignore; attn_mask is True at pairs to keep.
    # score_bias [n_query, n_key] is added to the attention scores.
    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, need_weights=False, score_bias=None):
        q, k, v = self._split(self.q(query)), self._split(self.k(key)), self._split(self.v(value))
        if self.qk_norm:
            # Unit-RMS heads cap |q.k|/sqrt(head_dim) at sqrt(head_dim). Without it, logits grew to 1e7 on XSum.
            q, k = F.rms_norm(q, (self.head_dim,)), F.rms_norm(k, (self.head_dim,))
        keep = attn_mask
        if key_padding_mask is not None:
            keep_keys = ~key_padding_mask[:, None, None, :]
            keep = keep_keys if keep is None else keep & keep_keys
        weights = None
        if need_weights:
            scores = q @ k.transpose(-1, -2) / self.head_dim ** 0.5
            if score_bias is not None:
                scores = scores + score_bias
            if keep is not None:
                scores = scores.masked_fill(~keep, torch.finfo(scores.dtype).min)
            probs = scores.softmax(-1)
            weights = probs.mean(1)
            y = probs @ v
        else:
            if score_bias is not None:
                bias = score_bias.to(q.dtype)
                keep = bias if keep is None else bias.masked_fill(~keep, float("-inf"))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=keep, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).reshape(query.shape[0], query.shape[1], -1)
        return self.out(y), weights


class MLP(nn.Sequential):
    def __init__(self, d, dropout=0.0, ratio=4):
        super().__init__(nn.Linear(d, ratio * d), nn.GELU(), nn.Linear(ratio * d, d), nn.Dropout(dropout))


class Block(nn.Module):
    def __init__(self, d, n_heads, dropout=0.0):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, n_heads, dropout)
        self.mlp = MLP(d, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        n = self.ln1(x)
        x = x + self.drop(self.attn(n, n, n, key_padding_mask, attn_mask)[0])
        return x + self.mlp(self.ln2(x))


class CrossBlock(nn.Module):
    def __init__(self, d, n_heads, dropout=0.0):
        super().__init__()
        self.ln1, self.ln2, self.ln_mem, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_attn = Attention(d, n_heads, dropout)
        self.cross_attn = Attention(d, n_heads, dropout)
        self.mlp = MLP(d, dropout)

    def forward(self, x, memory, attn_mask=None):
        n = self.ln1(x)
        x = x + self.self_attn(n, n, n, attn_mask=attn_mask)[0]
        m = self.ln_mem(memory)
        x = x + self.cross_attn(self.ln2(x), m, m)[0]
        return x + self.mlp(self.ln3(x))
