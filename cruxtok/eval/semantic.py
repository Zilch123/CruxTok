"""Phase 4 semantic evaluation: PAWS AUC and STS-B Spearman from code similarity at levels 1..k.

Evaluation only: PAWS and STS-B never reach training (see cruxtok/data/leakage.py).
Primary similarity: cosine between the mean codebook vectors of levels 1..k, the same function for every tokenizer.
"""
import argparse
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from cruxtok.data.text import fit_window, load_text_tokenizer
from cruxtok.model.layers import padding_positions

METRICS = {"paws": lambda labels, sims: roc_auc_score(labels, sims), "stsb": lambda labels, sims: spearmanr(labels, sims)[0]}


def load_pairs(task, split="test"):
    """Returns (sentences1, sentences2, labels). PAWS labels are 0/1, STS-B labels are scores in [0, 1]."""
    from datasets import load_dataset
    if task == "paws":
        ds = load_dataset("google-research-datasets/paws", "labeled_final", split=split)
        return list(ds["sentence1"]), list(ds["sentence2"]), np.array(ds["label"], dtype=float)
    ds = load_dataset("sentence-transformers/stsb", split=split)
    return list(ds["sentence1"]), list(ds["sentence2"]), np.array(ds["score"], dtype=float)


def load_run(path, device):
    """A finished run's tokenizer.pt, CruxTok or NCM, built from the config saved with it."""
    from cruxtok.training.tokenizer_trainer import build_tokenizer
    state = torch.load(os.path.join(path, "tokenizer.pt") if os.path.isdir(path) else path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(state["config"])
    tokenizer = build_tokenizer(cfg.model)
    tokenizer.load_state_dict(state["model"])
    return tokenizer.to(device).eval(), cfg


@torch.no_grad()
def level_embeddings(tokenizer, cfg, texts, max_k=3, batch_size=256):
    """[len(texts), max_k, d]: entry k-1 is the mean codebook vector over levels 1..k."""
    text_tok = load_text_tokenizer(cfg.model.text_tokenizer)
    L, eos = cfg.model.context_length, cfg.model.eos_id
    device = next(tokenizer.parameters()).device
    out = []
    for start in range(0, len(texts), batch_size):
        ids = torch.tensor([fit_window(text_tok.encode(t), L, eos) for t in texts[start:start + batch_size]], device=device)
        codes = tokenizer.encode_ids(ids)
        vectors = [tokenizer.code_vectors(codes, k).float() for k in range(max_k)]
        sums = torch.cumsum(torch.stack([v.sum(1) for v in vectors], 1), 1)
        counts = torch.cumsum(torch.tensor([v.shape[1] for v in vectors], device=device), 0)
        out.append((sums / counts[None, :, None]).cpu())
    return torch.cat(out)


@torch.no_grad()
def zhat_embeddings(tokenizer, cfg, texts, max_k=3, batch_size=256):
    """Secondary measure: [len(texts), max_k, d], the level-k reconstruction z_hat averaged over non-padding positions.
    The primary code mean also averages slots that pool EOS filler, so for short sentences it tracks length."""
    text_tok = load_text_tokenizer(cfg.model.text_tokenizer)
    L, eos, pad = cfg.model.context_length, cfg.model.eos_id, cfg.model.pad_id
    device = next(tokenizer.parameters()).device
    out = []
    for start in range(0, len(texts), batch_size):
        ids = torch.tensor([fit_window(text_tok.encode(t), L, eos) for t in texts[start:start + batch_size]], device=device)
        keep = (~padding_positions(ids, pad, eos)).float().unsqueeze(-1)
        z_hats = tokenizer.z_hats(tokenizer.encode_ids(ids))[:max_k]
        out.append(torch.stack([(z.float() * keep).sum(1) / keep.sum(1).clamp_min(1) for z in z_hats], 1).cpu())
    return torch.cat(out)


def code_similarities(tokenizer, cfg, s1, s2, max_k=3, embed=level_embeddings):
    """[n_pairs, max_k] cosine similarities, one column per k."""
    return F.cosine_similarity(embed(tokenizer, cfg, s1, max_k), embed(tokenizer, cfg, s2, max_k), dim=-1).numpy()


def jaccard_similarities(s1, s2):
    words = lambda t: set(re.findall(r"\w+", t.lower()))
    return np.array([len(words(a) & words(b)) / max(len(words(a) | words(b)), 1) for a, b in zip(s1, s2)])


def tfidf_similarities(s1, s2):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer().fit(s1 + s2)
    a, b = vec.transform(s1), vec.transform(s2)
    return np.asarray(a.multiply(b).sum(1)).ravel()  # rows are l2-normalized, so this is the cosine


@torch.no_grad()
def encoder_similarities(s1, s2, model_id="sentence-transformers/all-mpnet-base-v2", device="cpu", batch_size=64):
    """Ceiling: an off-the-shelf sentence encoder, mean-pooled as its model card specifies."""
    from transformers import AutoModel, AutoTokenizer
    tok, model = AutoTokenizer.from_pretrained(model_id), AutoModel.from_pretrained(model_id).to(device).eval()

    def embed(texts):
        out = []
        for start in range(0, len(texts), batch_size):
            batch = tok(texts[start:start + batch_size], padding=True, truncation=True, return_tensors="pt").to(device)
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            out.append(F.normalize((hidden * mask).sum(1) / mask.sum(1), dim=-1).cpu())
        return torch.cat(out)

    return (embed(s1) * embed(s2)).sum(-1).numpy()


def paired_bootstrap(task, labels, sims_a, sims_b, n=2000, seed=0):
    """Metric(a) - metric(b) on the same resampled pairs: point estimate and 95% CI."""
    metric, rng = METRICS[task], np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(labels), len(labels))
        diffs.append(metric(labels[idx], sims_a[idx]) - metric(labels[idx], sims_b[idx]))
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])  # a resample with one class has no AUC
    return {"diff": metric(labels, sims_a) - metric(labels, sims_b), "lo": float(lo), "hi": float(hi)}


def main():
    ap = argparse.ArgumentParser(description="PAWS AUC and STS-B Spearman from code similarity at k=1..3")
    ap.add_argument("--run", action="append", default=[], help="name=path to a run dir or tokenizer.pt; repeatable")
    ap.add_argument("--compare", action="append", default=[], help="a:b, paired bootstrap of run a minus run b; repeatable")
    ap.add_argument("--max_k", type=int, default=3)
    ap.add_argument("--no_ceiling", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="first n pairs per task, for quick checks")
    ap.add_argument("--out", default="semantic.json")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = dict(spec.split("=", 1) for spec in a.run)
    results = {}
    for task in ("paws", "stsb"):
        s1, s2, labels = load_pairs(task)
        if a.limit:
            s1, s2, labels = s1[:a.limit], s2[:a.limit], labels[:a.limit]
        metric = METRICS[task]
        sims = {"jaccard": jaccard_similarities(s1, s2), "tfidf": tfidf_similarities(s1, s2)}
        if not a.no_ceiling:
            sims["ceiling"] = encoder_similarities(s1, s2, device=device)
        rows = {name: {"k=all": float(metric(labels, s))} for name, s in sims.items()}
        rows["random"] = {"k=all": float(metric(labels, np.random.default_rng(0).random(len(labels))))}
        for name, path in runs.items():
            tokenizer, cfg = load_run(path, device)
            sims[name] = code_similarities(tokenizer, cfg, s1, s2, a.max_k)
            sims[f"{name}/zhat"] = code_similarities(tokenizer, cfg, s1, s2, a.max_k, embed=zhat_embeddings)
            for key in (name, f"{name}/zhat"):
                rows[key] = {f"k={k + 1}": float(metric(labels, sims[key][:, k])) for k in range(a.max_k)}
        comparisons = {}
        for spec in a.compare:
            x, y = spec.split(":")
            for suffix in ("", "/zhat"):
                comparisons[spec + suffix] = {f"k={k + 1}": paired_bootstrap(task, labels, sims[x + suffix][:, k], sims[y + suffix][:, k])
                                              for k in range(a.max_k)}
        results[task] = {"n_pairs": len(labels), "scores": rows, "comparisons": comparisons}
        print(f"\n{task} ({len(labels)} pairs, {'AUC' if task == 'paws' else 'Spearman'})")
        for name, r in rows.items():
            print(f"  {name:24s} " + "  ".join(f"{k} {v:.4f}" for k, v in r.items()))
        for spec, by_k in comparisons.items():
            print(f"  {spec:24s} " + "  ".join(f"{k} {c['diff']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]" for k, c in by_k.items()))
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
