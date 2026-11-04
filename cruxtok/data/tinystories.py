import numpy as np

from cruxtok.data.text import HF_DATASETS


def _fetch(name):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HF_DATASETS, f"tinystories/{name}", repo_type="dataset")


class TinyStoriesWindows:
    """Story openings from the released BPE stream, read in place so sampling stays exactly resumable."""

    def __init__(self, length, eos_id=0, train_bin=None, offsets=None, valid_bin=None):
        self.length, self.eos_id = length, eos_id
        self.train = np.memmap(train_bin or _fetch("train_bpe_8192.bin"), dtype=np.uint16, mode="r")
        self.offsets = np.load(offsets or _fetch("train_bpe_8192_offsets.npy"))
        self.valid = np.memmap(valid_bin or _fetch("valid_bpe_8192.bin"), dtype=np.uint16, mode="r")

    def _setup_valid(self, stream, length, eos_id):
        self.valid, self.length, self.eos_id = stream, length, eos_id
        return self

    def __len__(self):
        return len(self.offsets) - 1

    def window(self, i):
        return self._fill(self.train[self.offsets[i]:min(self.offsets[i + 1], self.offsets[i] + self.length)])

    def _fill(self, story):
        out = np.full(self.length, self.eos_id, dtype=np.int64)
        out[:len(story)] = story
        return out

    def sample(self, batch_size, rng):
        return np.stack([self.window(i) for i in rng.integers(0, len(self), batch_size)])

    def valid_windows(self, n):
        ends = np.flatnonzero(self.valid == self.eos_id)[:n] + 1
        starts = np.concatenate([[0], ends[:-1]])
        return np.stack([self._fill(self.valid[s:min(e, s + self.length)]) for s, e in zip(starts, ends)])

    def heldout(self, n):
        return self.valid_windows(n)
