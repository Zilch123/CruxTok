"""Structured random data, so smoke runs and tests need no downloads.

A window is a sequence of tokens drawn from a per-topic subset of the vocabulary, with varying length.
"""
import numpy as np

SPECIALS = 2  # eos=0, pad=1


class SyntheticWindows:
    def __init__(self, vocab_size, length, n_topics=16, topic_size=8, pad_id=1, eos_id=0, seed=0):
        self.length, self.pad_id, self.eos_id = length, pad_id, eos_id
        rng = np.random.default_rng(seed)
        self.topics = np.stack([rng.choice(np.arange(SPECIALS, vocab_size), topic_size, replace=False) for _ in range(n_topics)])

    def _make(self, n, rng):
        topic = rng.integers(0, len(self.topics), n)
        ids = self.topics[topic[:, None], rng.integers(0, self.topics.shape[1], (n, self.length))]
        lengths = rng.integers(self.length // 2, self.length + 1, n)
        ids[np.arange(self.length)[None] >= lengths[:, None]] = self.eos_id  # NCM's fill convention
        return ids.astype(np.int64), topic, lengths

    def sample(self, batch_size, rng):
        return self._make(batch_size, rng)[0]

    def heldout(self, n):
        return self._make(n, np.random.default_rng(12345))[0]
