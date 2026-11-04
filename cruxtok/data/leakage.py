"""PAWS and STS-B are evaluation-only. These checks fail a run if their text appears in training data."""
import hashlib
import re
import unicodedata


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def text_hash(text):
    return hashlib.sha1(normalize(text).encode("utf-8")).digest()


def units(text):
    """The whole text and each sentence in it, so a sentence embedded in a longer text is caught too."""
    yield text
    parts = re.split(r"(?<=[.!?])\s+", text)
    if len(parts) > 1:
        yield from parts


def eval_texts():
    from datasets import load_dataset
    for split in ("train", "validation", "test"):
        for row in load_dataset("google-research-datasets/paws", "labeled_final", split=split):
            yield row["sentence1"]
            yield row["sentence2"]
        for row in load_dataset("sentence-transformers/stsb", split=split):
            yield row["sentence1"]
            yield row["sentence2"]


def build_blocklist(texts=None):
    return {text_hash(t) for t in (eval_texts() if texts is None else texts)}


def find_leaks(texts, blocklist):
    return [t for t in texts if any(text_hash(u) in blocklist for u in units(t))]


def assert_no_leakage(texts, blocklist, name="training data"):
    leaks = find_leaks(texts, blocklist)
    assert not leaks, f"{len(leaks)} eval texts found in {name}; first: {leaks[0][:120]!r}"
