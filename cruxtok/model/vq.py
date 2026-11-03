import torch
import torch.nn as nn


class LevelCodebook(nn.Module):
    """One level's EMA codebook. Always runs in full precision: distances and EMA sums are not fp16-safe."""

    def __init__(self, size, d, decay=0.95, epsilon=1e-5, restart_threshold=0.01):
        super().__init__()
        self.size, self.decay, self.epsilon, self.restart_threshold = size, decay, epsilon, restart_threshold
        self.embedding = nn.Embedding(size, d)
        self.embedding.weight.data.normal_()
        self.embedding.weight.requires_grad_(False)
        self.register_buffer("ema_cluster_size", torch.zeros(size))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())

    def forward(self, u, update=False):
        with torch.autocast(u.device.type, enabled=False):
            flat = u.detach().float().reshape(-1, u.shape[-1])
            w = self.embedding.weight
            distances = flat.pow(2).sum(1, keepdim=True) + w.pow(2).sum(1) - 2 * flat @ w.t()
            idx = distances.argmin(1)
            if update:
                self._update(flat, idx)
            return self.embedding(idx).view(u.shape), idx.view(u.shape[:-1])

    @torch.no_grad()
    def _update(self, flat, idx):
        counts = torch.zeros_like(self.ema_cluster_size).index_add_(0, idx, torch.ones_like(idx, dtype=flat.dtype))
        dw = torch.zeros_like(self.ema_w).index_add_(0, idx, flat)
        self.ema_cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)
        n = self.ema_cluster_size.sum()
        smoothed = (self.ema_cluster_size + self.epsilon) / (n + self.size * self.epsilon) * n
        self.embedding.weight.data.copy_(self.ema_w / smoothed.unsqueeze(1))
        if self.restart_threshold > 0:
            self._restart(flat)

    # The threshold is relative to uniform usage. An absolute one would flag most codes at coarse levels,
    # where a batch makes far fewer assignments than there are codes.
    def _restart(self, flat):
        uniform = flat.shape[0] / self.size
        dead = self.ema_cluster_size < self.restart_threshold * uniform
        n_dead = int(dead.sum())
        if n_dead == 0:
            return
        new = flat[torch.randint(0, flat.shape[0], (n_dead,), device=flat.device)]
        self.embedding.weight.data[dead] = new
        self.ema_cluster_size[dead] = uniform
        self.ema_w[dead] = new * uniform

    @torch.no_grad()
    def perplexity(self):
        p = self.ema_cluster_size / self.ema_cluster_size.sum().clamp_min(1e-12)
        return torch.exp(-(p * p.clamp_min(1e-12).log()).sum()).item()
