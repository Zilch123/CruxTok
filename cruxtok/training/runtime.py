# Adapted from the reconstructed NCM trainer: https://github.com/Zilch123/ncm (branch sparse-level-supervision, train.py).
#
# MIT License
#
# Copyright (c) 2025 hermann kumbong
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import glob
import hashlib
import json
import math
import os
import random

import numpy as np
import torch
from omegaconf import OmegaConf


def load_config(path, overrides=()):
    cfg = OmegaConf.load(path)
    base = cfg.pop("base", None)
    if base:
        cfg = OmegaConf.merge(load_config(os.path.join(os.path.dirname(path), base)), cfg)
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def state_hash(module):
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def lr_at(step, t):
    warmup = min(t.warmup, t.steps // 10)
    if step < warmup:
        return t.lr * (step + 1) / warmup
    x = min((step - warmup) / max(t.steps - warmup, 1), 1.0)
    return t.lr_floor + 0.5 * (t.lr - t.lr_floor) * (1 + math.cos(math.pi * x))


def amp_dtype(precision):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[precision]


def latest_checkpoint(path):
    if os.path.isfile(path):
        return path
    found = sorted(glob.glob(os.path.join(path, "ckpt_*.pt")))
    return found[-1] if found else None


def save_checkpoint(run_dir, step, model, opt, scaler, data_rng, wandb_id, keep):
    state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(), "step": step,
             "wandb_id": wandb_id, "rng": {"python": random.getstate(), "numpy": data_rng.bit_generator.state,
                                           "torch": torch.get_rng_state(),
                                           "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}
    path = os.path.join(run_dir, f"ckpt_{step:07d}.pt")
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)
    for old in sorted(glob.glob(os.path.join(run_dir, "ckpt_*.pt")))[:-keep]:
        os.remove(old)
    return path


def load_checkpoint(path, model, opt, scaler, data_rng, device):
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    opt.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    random.setstate(state["rng"]["python"])
    data_rng.bit_generator.state = state["rng"]["numpy"]
    torch.set_rng_state(state["rng"]["torch"].cpu())
    if state["rng"]["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["rng"]["cuda"]])
    return state["step"], state["wandb_id"]


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def weight_decay_groups(module, weight_decay):
    # An attention prior is a log-probability table, so decay would erode its starting windows.
    named = [(n, p) for n, p in module.named_parameters() if p.requires_grad]
    decays = lambda n, p: p.dim() >= 2 and not n.endswith(".prior")
    return [{"params": [p for n, p in named if decays(n, p)], "weight_decay": weight_decay},
            {"params": [p for n, p in named if not decays(n, p)], "weight_decay": 0.0}]
