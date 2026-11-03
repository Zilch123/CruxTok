"""How far pooling attention moves from its starting windows, and whether it responds to the input.

If attention stays on the windows, CruxTok is close to NCM's fixed pooling, whatever the other metrics say.
"""
import argparse
import json
import os

import torch
from omegaconf import OmegaConf

from cruxtok.data.sources import build_data
from cruxtok.model.cruxtok import CruxTok
from cruxtok.model.layers import window_prior


def _moved(a, b):
    """Share of attention mass that differs, 0 to 1."""
    return 0.5 * (a - b).abs().sum(-1).mean().item()


def _text_only(w, text):
    """Restrict attention [B, l, L] to text positions [B, L] and renormalise."""
    w = w * text[:, None, :]
    return w / w.sum(-1, keepdim=True).clamp_min(1e-9)


@torch.no_grad()
def attention_report(tokenizer, ids, sharpness, eos_id=None):
    was_training = tokenizer.training
    tokenizer.eval()
    L = tokenizer.context_length
    # Text positions: up to and including the first EOS; NCM fills the rest of the window with EOS.
    text = None if eos_id is None else ((ids == eos_id).long().cumsum(-1) - (ids == eos_id).long() == 0).float()
    rows = []
    for k, w in enumerate(tokenizer.attention_maps(ids)):
        if w is None:
            continue
        pool = tokenizer.pools[k]
        start = window_prior(tokenizer.levels[k], L, sharpness).to(w.device).softmax(-1)
        prior = None if pool.prior is None else pool.prior.softmax(-1)
        if text is not None:
            # Movement on text tokens only, so attention leaving padding does not count as moving.
            wt, st = _text_only(w, text), _text_only(start.expand_as(w), text)
            padding = {"padding_share": 1 - text.mean().item(),
                       "mass_on_padding": (w * (1 - text)[:, None, :]).sum(-1).mean().item(),
                       "start_mass_on_padding": (start * (1 - text)[:, None, :]).sum(-1).mean().item(),
                       "attention_vs_start_text": 0.5 * (wt - st).abs().sum(-1).mean().item(),
                       "between_inputs_text": _moved(wt, wt.mean(0, keepdim=True))}
        rows.append({
            "level": k + 1,
            "slots": tokenizer.levels[k],
            "prior_vs_start": None if prior is None else _moved(prior, start),
            "attention_vs_start": _moved(w.mean(0), start),
            "between_inputs": _moved(w, w.mean(0, keepdim=True)),
            "width": torch.exp(-(w * w.clamp_min(1e-9).log()).sum(-1)).mean().item(),
            "start_width": torch.exp(-(start * start.clamp_min(1e-9).log()).sum(-1)).mean().item(),
            "top_token_spread_between_slots": w.argmax(-1).float().std(1).mean().item() if w.shape[1] > 1 else 0.0,
            **(padding if text is not None else {}),
        })
    tokenizer.train(was_training)
    return rows


def main():
    ap = argparse.ArgumentParser(description="report attention movement for a finished CruxTok run")
    ap.add_argument("--run", required=True, help="run directory holding tokenizer.pt")
    ap.add_argument("--n", type=int, default=256)
    a = ap.parse_args()
    state = torch.load(os.path.join(a.run, "tokenizer.pt"), map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(state["config"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = CruxTok(cfg.model).to(device)
    tokenizer.load_state_dict(state["model"])
    windows = build_data(cfg, cfg.seed)
    ids = torch.as_tensor(windows.heldout(a.n), device=device)
    rows = attention_report(tokenizer, ids, cfg.model.prior_sharpness, cfg.model.eos_id)
    with open(os.path.join(a.run, "attention.json"), "w") as f:
        json.dump(rows, f, indent=1)
    for r in rows:
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
