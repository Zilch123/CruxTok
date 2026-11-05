# Supplementary results

Outputs behind numbers in the paper that are not in the main `results/` files.

| File | Paper claim |
|---|---|
| `baselines.txt` | Table 5 baselines: word Jaccard, TF-IDF and the all-mpnet-base-v2 ceiling (PAWS 0.652, STS-B 0.834). `t1` is T1 seed 42. |
| `precision_fp16_vs_bf16.txt` | Table 1 caption: the fp16 (seed 42) and bf16 (seed 2) T0-cap runs do not differ on PAWS or STS-B at any level (every paired-bootstrap CI includes 0). |
| `judge_validation_*.json` | Section 5: the three other judge candidates scored on STS-B. Kimi-K2.5 parsed 852 of 1,379 items (38% empty) and was excluded. The chosen judge, GLM-5.3, is in `results/`. The runner-up, Qwen3-235B-Instruct-2507, is the second judge in Section 5; its results are in `results/judge_qwen/`. |
| `prior_sharpness_pilots.md` | Section 4: the prior sharpness of 1.0 was chosen on held-out reconstruction in 4,000-step pilots. `ts` is TinyStories, `xs` is XSum. On XSum, sharpness 1 beats 4 at k=1 and k=3 on text tokens. These pilots used shorter training and are not the reported runs. |
| `stsb_length_and_params.json` | Section 5: on STS-B the primary measure correlates with the shorter sentence's GPT-2 length; also each tokenizer's parameter count (Table 1). Recomputed on CPU from the Hugging Face checkpoints: ρ is −0.20 to −0.71 across all five at k=1–3 (T1: −0.63 to −0.71). Parameter counts: T0-cap 60.63M, matched 73.06M, T1 73.23M. |
| `attention_example.json` | Figure 2b: level-1 attention of T1 seed 42 on validation summary 197, the first of a random draw (seed 0) from the 256 held-out windows, with the starting prior. Made by `scripts/attention_example.py`. |
| `../matched_s2/` | Section 6: the second seed of the matched T0-cap (seed 2, bf16). Judges (GLM-5.3 and `judge_qwen/`) for T1 seeds 42 and 2 against it at k=1-3, embedding cosine (`embedding.json`, which also compares the two matched seeds), PAWS and STS-B (`code_similarity.*`), and its reconstructions of the 1,000 test summaries. |
