import math

import torch
import torch.nn.functional as F

from cruxtok.model.layers import padding_positions


@torch.no_grad()
def evaluate_reconstruction(tokenizer, ids, batch_size=64, autocast=None):
    """Exact token accuracy after each level, and per-level codebook use. Depends only on the tokenizer interface."""
    was_training = tokenizer.training
    tokenizer.eval()
    device = next(tokenizer.parameters()).device
    K = len(tokenizer.levels)
    hits, hits_text, ce = torch.zeros(K), torch.zeros(K), torch.zeros(K)
    counts = [torch.zeros(v, device=device) for v in tokenizer.vocab_sizes]
    n_tokens, n_text = 0, 0
    for start in range(0, ids.shape[0], batch_size):
        x = torch.as_tensor(ids[start:start + batch_size], device=device)
        codes = tokenizer.encode_ids(x)
        for k, level_codes in enumerate(tokenizer.split_levels(codes)):
            counts[k].index_add_(0, level_codes.reshape(-1), torch.ones(level_codes.numel(), device=device))
        is_text = ~padding_positions(x, tokenizer.pad_id, tokenizer.eos_id)
        n_tokens += x.numel()
        n_text += int(is_text.sum())
        for k, z_hat in enumerate(tokenizer.z_hats(codes)):
            logits = tokenizer.decode_logits(z_hat).float()
            right = logits.argmax(-1) == x
            hits[k] += right.sum().item()
            hits_text[k] += (right & is_text).sum().item()
            ce[k] += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x.reshape(-1), reduction="sum").item()
    tokenizer.train(was_training)
    used, perplexity = [], []
    for c in counts:
        p = c / c.sum()
        used.append(100.0 * (c > 0).float().mean().item())
        perplexity.append(math.exp(-(p * p.clamp_min(1e-12).log()).sum().item()))
    return {
        "accuracy": (100.0 * hits / n_tokens).tolist(),
        "accuracy_text": (100.0 * hits_text / max(n_text, 1)).tolist(),
        "ce": (ce / n_tokens).tolist(),
        "utilization": used,
        "perplexity": perplexity,
    }
