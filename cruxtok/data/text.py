import io
import pickle

HF_DATASETS = "hazyresearch/ncm-datasets"


class _NoGlobals(pickle.Unpickler):
    """The BPE files hold only dicts, lists and bytes, so nothing they load needs a global."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def _load_pickle(path):
    with open(path, "rb") as f:
        return _NoGlobals(io.BytesIO(f.read())).load()


class BPEText:
    eos_id, pad_id = 0, 1

    def __init__(self):
        from huggingface_hub import hf_hub_download
        from cruxtok.ncm import ensure_path
        ensure_path()
        from tokenization.bpe_tokenizer import BPETokenizer
        fetch = lambda name: _load_pickle(hf_hub_download(HF_DATASETS, f"tinystories/{name}", repo_type="dataset"))
        self.bpe = BPETokenizer(fetch("bpe_vocab_8192.pickle"), fetch("bpe_merges_8192.pickle"),
                                {"eos_token": "<|endoftext|>", "pad_token": "<|pad|>"})

    def encode(self, text):
        return self.bpe.encode(text)

    def decode(self, ids):
        return self.bpe.decode([int(i) for i in ids], skip_special_tokens=True)


class GPT2Text:
    eos_id, pad_id = 50256, 50257

    def __init__(self):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained("gpt2")

    def encode(self, text):
        return self.tok(text)["input_ids"]

    def decode(self, ids):
        return self.tok.decode([int(i) for i in ids if int(i) < self.eos_id])


class ByteText:
    """Needs no download; for tests."""
    eos_id, pad_id = 0, 1

    def encode(self, text):
        return [b + 2 for b in text.encode("utf-8")]

    def decode(self, ids):
        return bytes(int(i) - 2 for i in ids if int(i) >= 2).decode("utf-8", errors="replace")


def load_text_tokenizer(name):
    return {"bpe": BPEText, "gpt2": GPT2Text, "bytes": ByteText}[name]()


def fit_window(ids, length, eos_id, add_eos=True):
    """A window of exactly `length` ids, filled with EOS as NCM does."""
    ids = (list(ids) + ([eos_id] if add_eos else []))[:length]
    return ids + [eos_id] * (length - len(ids))
