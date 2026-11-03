"""How much meaning can be decoded from levels 1..k alone.

Measure A: embedding cosine between an original and its level-k reconstruction.
Measure B: a pairwise LLM judge, validated first on STS-B human scores (evaluation-only use).
LLM calls go to Nscale's OpenAI-compatible API, with the key read from NSCALE_API_KEY.
"""
import argparse
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import torch
from scipy.stats import spearmanr

from cruxtok.data.text import fit_window, load_text_tokenizer
from cruxtok.eval.semantic import load_pairs, load_run

API = "https://inference.api.nscale.com/v1"
EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"
# GLM-5.3 ignores enable_thinking on long prompts and reasoned for ~1000 tokens a call; reasoning_effort=low keeps it near 40.
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "low"}
# Spend guard from the token counts each response reports, at list prices (USD per 1M tokens, input / output).
PRICES = {"zai-org/GLM-5.3": (1.4, 4.4)}
MAX_COST = float(os.environ.get("JUDGE_MAX_COST", 4.5))
_spent = [0.0]

SCORE_PROMPT = ("How similar in meaning are these two sentences? Use the STS scale: 0 = unrelated, 5 = same meaning.\n"
                "Sentence 1: {a}\nSentence 2: {b}\nReply with only the number, for example 3.2.")
PAIR_PROMPT = ("An original sentence and two damaged reconstructions of it follow. Which reconstruction keeps more of the "
               "original's meaning: who, what, where, and the outcome? Ignore grammar, fluency, repetition and wording; "
               "judge only the meaning that is recoverable.\nOriginal: {o}\nReconstruction A: {a}\nReconstruction B: {b}\n"
               "Reply with only one letter: A, B, or T if they keep the same amount.")


def _headers():
    return {"Authorization": f"Bearer {os.environ['NSCALE_API_KEY']}"}


def chat(model, prompt, max_tokens=1500, retries=4):
    if _spent[0] > MAX_COST:
        raise RuntimeError(f"spend guard: ${_spent[0]:.2f} used, limit ${MAX_COST}")
    for attempt in range(retries):
        try:
            r = requests.post(f"{API}/chat/completions", headers=_headers(), timeout=180, json={
                "model": model, "temperature": 0, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}], **NO_THINKING})
            j = r.json()
            usage = j.get("usage", {})
            price_in, price_out = PRICES.get(model, (0.0, 0.0))
            _spent[0] += (usage.get("prompt_tokens", 0) * price_in + usage.get("completion_tokens", 0) * price_out) / 1e6
            return (j["choices"][0]["message"].get("content") or "").strip(), usage
        except Exception:
            if attempt == retries - 1:
                return "", {}


def map_chat(model, prompts, workers=int(os.environ.get("JUDGE_WORKERS", 32))):
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda p: chat(model, p), prompts))


def embed(texts, batch_size=64, workers=8):
    def one(start):
        batch = [t if t.strip() else "." for t in texts[start:start + batch_size]]
        j = requests.post(f"{API}/embeddings", headers=_headers(), timeout=180, json={"model": EMBED_MODEL, "input": batch}).json()
        return [d["embedding"] for d in sorted(j["data"], key=lambda d: d["index"])]
    with ThreadPoolExecutor(workers) as pool:
        out = [e for chunk in pool.map(one, range(0, len(texts), batch_size)) for e in chunk]
    v = np.array(out, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@torch.no_grad()
def reconstruct(tokenizer, cfg, texts, ks, batch_size=128):
    """{k: [text decoded from levels 1..k]}; k equal to the number of levels means full depth."""
    text_tok = load_text_tokenizer(cfg.model.text_tokenizer)
    L, eos = cfg.model.context_length, cfg.model.eos_id
    device = next(tokenizer.parameters()).device
    out = {k: [] for k in ks}
    for start in range(0, len(texts), batch_size):
        ids = torch.tensor([fit_window(text_tok.encode(t), L, eos) for t in texts[start:start + batch_size]], device=device)
        z_hats = tokenizer.z_hats(tokenizer.encode_ids(ids))
        for k in ks:
            for row in tokenizer.decode_logits(z_hats[k - 1]).argmax(-1).tolist():
                out[k].append(text_tok.decode(row[:row.index(eos)] if eos in row else row))
    return out


def parse_score(text):
    m = re.search(r"\d+(\.\d+)?", text)
    return float(m.group()) if m else np.nan


def parse_choice(text):
    """A, B or T; None when the call failed or the answer is unusable, so failures are never counted as ties."""
    m = re.search(r"\b([ABT])\b", (text or "").upper())
    return m.group(1) if m else None


def validate_judge(model, n_consistency=200, seed=0):
    """Spearman with STS-B human scores, plus order consistency on pairwise items."""
    s1, s2, labels = load_pairs("stsb")
    scores = np.array([parse_score(t) for t, _ in map_chat(model, [SCORE_PROMPT.format(a=a, b=b) for a, b in zip(s1, s2)])])
    ok = ~np.isnan(scores)
    rho = spearmanr(labels[ok], scores[ok])[0]
    # Consistency: original = sentence 1, candidates = sentence 2 of two random pairs, asked in both orders.
    rng = random.Random(seed)
    items = [(s1[i], s2[i], s2[j]) for i, j in (rng.sample(range(len(s1)), 2) for _ in range(n_consistency))]
    fwd = [parse_choice(t) for t, _ in map_chat(model, [PAIR_PROMPT.format(o=o, a=a, b=b) for o, a, b in items])]
    rev = [parse_choice(t) for t, _ in map_chat(model, [PAIR_PROMPT.format(o=o, a=b, b=a) for o, a, b in items])]
    flip = {"A": "B", "B": "A", "T": "T"}
    answered = [(f, r) for f, r in zip(fwd, rev) if f is not None and r is not None]
    consistent = float(np.mean([f == flip[r] for f, r in answered])) if answered else 0.0
    return {"model": model, "spearman": float(rho), "parsed": int(ok.sum()), "n": len(labels), "order_consistency": consistent,
            "consistency_answered": len(answered)}


def pairwise(model, originals, recon_a, recon_b, n_boot=2000, seed=0):
    """Win rate of a over b; a side wins only if both orders agree. Bootstrap CI over items."""
    items = list(zip(originals, recon_a, recon_b))
    prompts = [PAIR_PROMPT.format(o=o, a=a, b=b) for o, a, b in items] + [PAIR_PROMPT.format(o=o, a=b, b=a) for o, a, b in items]
    choices = [parse_choice(t) for t, _ in map_chat(model, prompts)]
    fwd, rev = choices[:len(items)], choices[len(items):]
    valid = [(f, r) for f, r in zip(fwd, rev) if f is not None and r is not None]
    if len(valid) < 0.95 * len(items):
        raise RuntimeError(f"only {len(valid)}/{len(items)} items have usable answers in both orders; not scoring")
    score = np.array([1.0 if (f, r) == ("A", "B") else 0.0 if (f, r) == ("B", "A") else 0.5 for f, r in valid])
    rng = np.random.default_rng(seed)
    boots = [score[rng.integers(0, len(score), len(score))].mean() for _ in range(n_boot)]
    return {"win_rate": float(score.mean()), "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)),
            "a_wins": int((score == 1).sum()), "b_wins": int((score == 0).sum()), "ties": int((score == 0.5).sum()), "n_valid": len(valid)}


def main():
    ap = argparse.ArgumentParser(description="meaning-recovery evaluation: embedding similarity and a pairwise LLM judge")
    ap.add_argument("step", choices=["validate", "reconstruct", "embed", "judge"])
    ap.add_argument("--model", action="append", default=[], help="judge model id; repeatable for validate")
    ap.add_argument("--run", action="append", default=[], help="name=path to a run dir")
    ap.add_argument("--compare", action="append", default=[], help="a:b pairwise comparison")
    ap.add_argument("--k", action="append", default=[], help="judge only these levels, e.g. k=1; default k=1..3")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default="runs/eval/meaning")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    recon_path = os.path.join(a.out, "reconstructions.json")

    if a.step == "validate":
        for m in a.model:
            r = validate_judge(m)
            r["settings"], r["spent_usd"] = NO_THINKING, round(_spent[0], 3)
            print(json.dumps(r), flush=True)
            json.dump(r, open(os.path.join(a.out, f"judge_validation_{m.replace('/', '_')}.json"), "w"), indent=1)
        return

    if a.step == "reconstruct":
        from datasets import load_dataset
        summaries = load_dataset("EdinburghNLP/xsum", split="test")["summary"]
        idx = sorted(random.Random(0).sample(range(len(summaries)), a.n))
        data = {"originals": [summaries[i] for i in idx], "runs": {}}
        for spec in a.run:
            name, path = spec.split("=", 1)
            tokenizer, cfg = load_run(path, "cuda" if torch.cuda.is_available() else "cpu")
            K = len(cfg.model.levels)
            data["runs"][name] = {f"k={k}" if k < K else "full": v for k, v in reconstruct(tokenizer, cfg, data["originals"], [1, 2, 3, K]).items()}
        json.dump(data, open(recon_path, "w"), indent=1)
        print(f"wrote {recon_path}")
        return

    data = json.load(open(recon_path))
    if a.step == "embed":
        orig = embed(data["originals"])
        cos = {name: {k: (embed(texts) * orig).sum(1) for k, texts in by_k.items()} for name, by_k in data["runs"].items()}
        for name, by_k in cos.items():
            print(name, "  ".join(f"{k} {v.mean():.3f}" for k, v in by_k.items()), flush=True)
        diffs = {}
        rng = np.random.default_rng(0)
        for spec in a.compare:
            x, y = spec.split(":")
            diffs[spec] = {}
            for k in cos[x]:
                d = cos[x][k] - cos[y][k]
                boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
                diffs[spec][k] = {"diff": float(d.mean()), "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5))}
            print(spec, "  ".join(f"{k} {r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]" for k, r in diffs[spec].items()), flush=True)
        json.dump({"model": EMBED_MODEL, "per_item": {n: {k: v.tolist() for k, v in b.items()} for n, b in cos.items()}, "comparisons": diffs},
                  open(os.path.join(a.out, "embedding.json"), "w"))
        return

    results = {}
    ks = a.k or ["k=1", "k=2", "k=3"]
    for spec in a.compare:
        x, y = spec.split(":")
        results[spec] = {k: pairwise(a.model[0], data["originals"], data["runs"][x][k], data["runs"][y][k]) for k in ks}
        for k, r in results[spec].items():
            print(f"{spec} {k}: win rate {r['win_rate']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}] (wins {r['a_wins']}, losses {r['b_wins']}, ties {r['ties']})", flush=True)
        print(f"spent so far ${_spent[0]:.2f}", flush=True)
        json.dump({"judge": a.model[0], "settings": NO_THINKING, "spent_usd": round(_spent[0], 3), "results": results},
                  open(os.path.join(a.out, f"judge_{'_'.join(s.replace(':', '-') for s in a.compare)}{'_' + '_'.join(a.k) if a.k else ''}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
