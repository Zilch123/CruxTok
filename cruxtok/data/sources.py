from cruxtok.data.synthetic import SyntheticWindows


def build_data(cfg, seed):
    """Training windows, with sample(B, rng) and heldout(n)."""
    d, m = cfg.data, cfg.model
    if d.dataset == "synthetic":
        return SyntheticWindows(m.in_vocab_size, m.context_length, pad_id=m.pad_id, eos_id=m.eos_id, seed=seed)
    if d.dataset == "tinystories":
        from cruxtok.data.tinystories import TinyStoriesWindows
        return TinyStoriesWindows(m.context_length, m.eos_id)
    if d.dataset == "xsum":
        from cruxtok.data.xsum import XSumWindows
        return XSumWindows(m.context_length, d.root)
    raise ValueError(f"unknown dataset {d.dataset}")
