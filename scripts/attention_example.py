"""Level-1 pooling attention of T1 seed 42 on one held-out XSum validation summary (drawn at random, seed 0),
and the starting prior. Writes pgfplots coordinates. Run from the repo root: python scripts/attention_example.py list 6, then python scripts/attention_example.py 197"""
import json
import random
import sys

import torch
from huggingface_hub import hf_hub_download

from cruxtok.data.text import fit_window, load_text_tokenizer
from cruxtok.data.xsum import load_summaries
from cruxtok.eval.semantic import load_run
from cruxtok.model.layers import window_prior

tokenizer, cfg = load_run(hf_hub_download("TimothDevA/cruxtok", "t1_seed42/tokenizer.pt"), "cpu")
L, eos = cfg.model.context_length, cfg.model.eos_id
summaries = load_summaries("validation")[:256]  # the 256 held-out windows the attention report uses
draws = random.Random(0).sample(range(256), int(sys.argv[2]) if len(sys.argv) > 2 else 5)
mode = sys.argv[1]
if mode == "list":
    for i in draws:
        print(i, summaries[i])
    sys.exit()

i = int(mode)
tok = load_text_tokenizer("gpt2")
enc = tok.encode(summaries[i])
ids = torch.tensor([fit_window(enc, L, eos)])
w = tokenizer.attention_maps(ids)[0][0]  # [4 slots, 48], mean over heads
start = window_prior(tokenizer.levels[0], L, cfg.model.prior_sharpness).softmax(-1)
n_text = len(enc) + 1  # text plus the first EOS
out = {"index": i, "summary": summaries[i], "n_text": n_text,
       "tokens": [tok.tok.decode([t]) for t in enc],
       "attention": [[round(float(x), 4) for x in row] for row in w],
       "start_prior": [[round(float(x), 4) for x in row] for row in start],
       "padding_mass": float(w[:, n_text:].sum(-1).mean())}
json.dump(out, open("results/supplementary/attention_example.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("index", "summary", "n_text", "tokens", "padding_mass")}))
print("argmax per slot", w.argmax(-1).tolist(), "eff tokens", torch.exp(-(w * w.clamp_min(1e-9).log()).sum(-1)).tolist())
