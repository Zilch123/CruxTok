"""STS-B test: Spearman between the primary code similarity (k=1..3) and the shorter sentence's GPT-2 length,
plus each tokenizer's parameter count as the trainer prints it. Run from the repo root: python scripts/stsb_length_and_params.py results/supplementary/stsb_length_and_params.json"""
import json
import sys

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from scipy.stats import spearmanr

from cruxtok.data.text import load_text_tokenizer
from cruxtok.eval.semantic import code_similarities, load_pairs, load_run

RUNS = ["t0cap_seed42", "t0cap_seed2", "t0cap_matched", "t1_seed42", "t1_seed2"]
s1, s2, _ = load_pairs("stsb")
tok = load_text_tokenizer("gpt2")
shorter = np.array([min(len(tok.encode(a)), len(tok.encode(b))) for a, b in zip(s1, s2)])
out = {"task": "stsb test", "n_pairs": len(s1), "length": "shorter sentence, GPT-2 tokens", "runs": {}}
for name in RUNS:
    tokenizer, cfg = load_run(hf_hub_download("TimothDevA/cruxtok", f"{name}/tokenizer.pt"), "cpu")
    sims = code_similarities(tokenizer, cfg, s1, s2, 3)
    params = tokenizer.num_params() if hasattr(tokenizer, "num_params") else sum(p.numel() for p in tokenizer.parameters())
    out["runs"][name] = {"params_M": round(params / 1e6, 2),
                         "spearman_with_length": {f"k={k + 1}": round(float(spearmanr(sims[:, k], shorter)[0]), 3) for k in range(3)}}
    print(name, out["runs"][name], flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
