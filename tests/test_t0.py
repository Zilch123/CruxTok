"""Released tokenizers (T0). Run with: pytest -m slow. Downloads about 520 MB."""
import numpy as np
import pytest

from cruxtok.eval.recon import evaluate_reconstruction
from cruxtok.ncm import load_released

N_WINDOWS = 512


def tinystories_valid_windows(n, length, eos_id=0):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("hazyresearch/ncm-datasets", "tinystories/valid_bpe_8192.bin", repo_type="dataset")
    stream = np.memmap(path, dtype=np.uint16, mode="r")
    from cruxtok.data.tinystories import TinyStoriesWindows
    return TinyStoriesWindows.__new__(TinyStoriesWindows)._setup_valid(stream, length, eos_id).valid_windows(n)


def xsum_valid_windows(n, length, eos_id):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    texts = load_dataset("EdinburghNLP/xsum", split=f"validation[:{n}]")["summary"]
    windows = np.full((n, length), eos_id, dtype=np.int64)
    for row, text in zip(windows, texts):
        ids = (tok(text)["input_ids"] + [tok.eos_token_id])[:length]
        row[:len(ids)] = ids
    return windows


@pytest.mark.slow
@pytest.mark.parametrize("task,paper_acc,paper_util", [("tinystories", 99.98, 90.12), ("xsum", 99.79, 86.20)])
def test_t0_reconstructs_released_accuracy(task, paper_acc, paper_util):
    tok, _ = load_released(task)
    assert tok.levels[0] == (1 if task == "tinystories" else 4)
    if task == "tinystories":
        ids = tinystories_valid_windows(N_WINDOWS, tok.context_length, tok.eos_id)
    else:
        ids = xsum_valid_windows(N_WINDOWS, tok.context_length, tok.eos_id)
    r = evaluate_reconstruction(tok, ids, 32)
    print(task, "levels", tok.levels)
    print(task, "params M", tok.num_params() / 1e6)
    print(task, "accuracy per level", [round(x, 2) for x in r["accuracy"]])
    print(task, "accuracy text only", [round(x, 2) for x in r["accuracy_text"]])
    print(task, "utilization on this sample", [round(x, 1) for x in r["utilization"]], "paper mean", paper_util)
    assert abs(r["accuracy_text"][-1] - paper_acc) < 0.3, (r["accuracy_text"][-1], paper_acc)
    assert r["accuracy"][-1] > 99.0, "the EOS fill must reconstruct too: it is the convention T0 trained with"
    assert r["accuracy"][-1] > r["accuracy"][0], "accuracy must rise with depth"
