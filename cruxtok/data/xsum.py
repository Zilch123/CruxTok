import os

import numpy as np

from cruxtok.data.text import fit_window, load_text_tokenizer


def load_summaries(split):
    from datasets import load_dataset
    return load_dataset("EdinburghNLP/xsum", split=split)["summary"]


class XSumWindows:
    """XSum target sentences under GPT-2. Tokenized once and cached under root."""

    def __init__(self, length, root="data"):
        self.length, self.root = length, root
        self.train = self._tokenized("train")

    def _tokenized(self, split):
        path = os.path.join(self.root, "xsum", f"{split}_{self.length}.npy")
        if os.path.exists(path):
            return np.load(path)
        tok = load_text_tokenizer("gpt2")
        ids = np.array([fit_window(tok.encode(t), self.length, tok.eos_id) for t in load_summaries(split)], dtype=np.int64)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, ids)
        return ids

    def sample(self, batch_size, rng):
        return self.train[rng.integers(0, len(self.train), batch_size)]

    def heldout(self, n):
        return self._tokenized("validation")[:n]
