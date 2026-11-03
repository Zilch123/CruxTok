# CruxTok

Learned-query pooling for the residual VQ text tokenizer of [Next-Concept Modeling](https://github.com/HazyResearch/ncm) (NCM). With the same parameter count as an NCM baseline, CruxTok's first level recovers much more of a sentence.

Paper: [What Does the First Level of a Residual Text Tokenizer Know? Learned-Query Pooling Puts the Gist in Level 1](paper/main.pdf) (arXiv link to follow).

## Results
All runs are trained on XSum summaries for 30,000 updates at batch 256. "k" means levels 1..k together: k=1 is 4 codes, k=3 is 20, and "full" is all 9 levels.

| | Params | Text accuracy, k=1 / k=3 / full | Embedding cosine to original, k=1 | LLM judge at k=1, T1 win rate |
|---|---|---|---|---|
| T0-cap (NCM, same budget) | 60.6M | 23.5 / 47.1 / 95.8 | 0.359 | 0.700 [0.678, 0.722] |
| T0-cap, matched (5 encoder / 3 decoder blocks) | 73.1M | 19.3 / 45.6 / 97.8 | 0.342 | 0.740 [0.720, 0.760] (seed 42), 0.772 [0.753, 0.792] (seed 2) |
| **T1, CruxTok** (seeds 42 / 2) | 73.2M | 23.5 / 60.1 / 98.8, 24.4 / 60.7 / 98.8 | 0.449 / 0.455 | |

- **Level 1 keeps the gist.** At k=1, CruxTok's reconstructions are about 0.1 closer to the original in embedding space than both baselines, on both seeds. A pairwise LLM judge, validated on STS-B (Spearman 0.90), prefers them in 70-77% of comparisons.
- **The queries choose tokens.** On text tokens, 37-53% of pooling attention moves away from NCM's starting windows, and it differs between inputs.
- **The limit.** CruxTok does not make similar sentences get similar codes. On STS-B at k=3, code similarity reaches Spearman 0.28 for the matched baseline against 0.02-0.03 for CruxTok.

Evaluation outputs behind these numbers are in [results/](results/).

## Install
```bash
git clone --recurse-submodules https://github.com/Zilch123/CruxTok
cd CruxTok
pip install -e ".[dev]"
pytest                    # fast tests, CPU
pytest -m slow            # released NCM tokenizers against the NCM paper; downloads about 520 MB
```

## Use a trained tokenizer
```python
from cruxtok.model.cruxtok import CruxTok

tok = CruxTok.from_pretrained("TimothDevA/cruxtok", filename="t1_seed42/tokenizer.pt")
codes = tok.encode("West Ham striker Andy Carroll is facing up to six weeks out of action.")
print(tok.decode(codes, upto_level=1))               # from the first 4 codes only
print(tok.decode(codes))                             # all levels
```
Trained tokenizers (CruxTok seeds 42 and 2, and the three NCM baselines): [huggingface.co/TimothDevA/cruxtok](https://huggingface.co/TimothDevA/cruxtok). `from_pretrained` also accepts a local run directory.

## Reproduce
Training (one GPU each; the reported runs used bf16, except T0-cap seed 42 in fp16):
```bash
python -m cruxtok.cli.train_tokenizer --config configs/xsum/t1.yaml runtime.precision=bf16 seed=42 run_name=xsum_t1
python -m cruxtok.cli.train_tokenizer --config configs/xsum/t1.yaml runtime.precision=bf16 seed=2 run_name=xsum_t1_s2
python -m cruxtok.cli.train_tokenizer --config configs/xsum/t0cap.yaml runtime.precision=fp16 run_name=xsum_t0cap
python -m cruxtok.cli.train_tokenizer --config configs/xsum/t0cap_matched.yaml runtime.precision=bf16
```
Overrides use dotted keys. `--benchmark 30` measures speed and memory, and `--resume_from <dir>` continues a run.

Evaluation:
```bash
# PAWS and STS-B code similarity, with a paired bootstrap (evaluation-only data)
python -m cruxtok.eval.semantic --run t0cap=runs/xsum_t0cap --run t1=runs/xsum_t1 --compare t1:t0cap
# Pooling attention movement
python -m cruxtok.eval.attention --run runs/xsum_t1
# Decodable meaning: reconstructions, embedding similarity, LLM judge (needs NSCALE_API_KEY)
python -m cruxtok.eval.meaning reconstruct --run t0cap=runs/xsum_t0cap --run t1=runs/xsum_t1
python -m cruxtok.eval.meaning embed --compare t1:t0cap
python -m cruxtok.eval.meaning validate --model zai-org/GLM-5.3
python -m cruxtok.eval.meaning judge --model zai-org/GLM-5.3 --compare t1:t0cap --k k=1
```
`JUDGE_MAX_COST` caps spend per process at list prices.

## Attribution
Built on [Next-Concept Modeling](https://github.com/HazyResearch/ncm) (Kumbong et al., arXiv:2610.14162, MIT), included unmodified as a submodule at `third_party/ncm`. Code adapted from it keeps its MIT header and an attribution line.

## License
MIT
