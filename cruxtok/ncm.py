"""Access to NCM's released code in third_party/ncm, which is never modified.

NCM imports itself through the top-level names `models`, `utils` and `tokenization`,
so every use goes through this module.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from cruxtok.model.cruxtok import TokenizerOutput

ROOT = os.environ.get("NCM_ROOT") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "ncm")
HF_CHECKPOINTS = "hazyresearch/ncm-checkpoints"


def ensure_path():
    if not os.path.isfile(os.path.join(ROOT, "models", "vqvae.py")):
        raise RuntimeError(f"NCM not found at {ROOT}; run: git submodule update --init")
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)


def vqvae_kwargs(m):
    """Maps our model config to NCM's VQVAE arguments; defaults match the released configs."""
    return OmegaConf.create(dict(
        n_layers=m.n_heads, embed_dim=m.d, in_vocab_size=m.in_vocab_size, context_length=m.context_length,
        compression_factor=1, dropout=m.dropout, pre_quant_groupnorm=8, use_qk_norm=True,
        pre_quant_dropout=m.get("pre_quant_dropout", 0.0), enc_layers=m.encoder_blocks, dec_layers=m.decoder_blocks,
        vqvae_vocab_size=[m.codebook_size] * len(m.levels),
        vector_quantizer_config=dict(
            clss="multiscale_residual_vector_quantizer", decay=m.vq.decay, epsilon=m.vq.epsilon, commitment_cost=m.vq.commitment,
            learned_l1_sampling=True, learned_l1_cascade=True, learned_l1_cascade_base=4, learned_all_sampling=False,
            learned_l1_cascade_norm=m.get("cascade_norm", False), learned_l1_cascade_norm_final=m.get("cascade_norm", False),
            quant_resi=dict(enabled=True, ratio=0.5, share_mode=0, learnable_ratio=False),
            aux=dict(fine_drop=dict(prob=m.nested_dropout.prob, min_keep=m.nested_dropout.min_keep)),
            levels=dict(use_manual_levels=True, manual_levels=list(m.levels)),
        ),
    ))


class NCMTokenizer(nn.Module):
    """T0 and T0-cap behind the same interface as CruxTok."""

    def __init__(self, vqvae, pad_id, eos_id):
        super().__init__()
        self.vqvae = vqvae
        self.vq = vqvae.vq
        self.levels = [int(l) for l in self.vq.levels]
        self.K = len(self.levels)
        self.vocab_sizes = list(self.vq.vqvae_vocab_sizes)
        self.embed_dim, self.context_length, self.in_vocab_size = vqvae.embed_dim, vqvae.context_length, vqvae.in_vocab_size
        self.pad_id, self.eos_id = pad_id, eos_id

    @classmethod
    def from_model_config(cls, m):
        ensure_path()
        from models.vqvae import VQVAE
        return cls(VQVAE(**vqvae_kwargs(m)), m.pad_id, m.eos_id)

    @property
    def embeddings(self):
        return list(self.vq.embeddings)

    # Quantization runs outside autocast: NCM's matmul would otherwise compute distances in fp16.
    def forward(self, ids, cutoff=None):
        z = self.vqvae._encode_pre_quant(ids)
        with torch.autocast(ids.device.type, enabled=False):
            z_q, vq_loss, *_ = self.vq(z.float())
        logits = self.vqvae.decoder(z_q)
        return TokenizerOutput(logits, vq_loss, None, -1, [self._perplexity(k) for k in range(self.K)])

    @torch.no_grad()
    def _perplexity(self, k):
        cs = getattr(self.vq, f"ema_cluster_size_{k}")
        p = cs / cs.sum().clamp_min(1e-12)
        return torch.exp(-(p * p.clamp_min(1e-12).log()).sum()).item()

    @torch.no_grad()
    def encode_ids(self, ids, attention_mask=None):
        return self.vqvae.encode(ids)  # NCM has no padding mask

    def split_levels(self, codes):
        return list(codes.split(self.levels, dim=1))

    def code_vectors(self, codes, k):
        return self.vq.embeddings[k](self.split_levels(codes)[k])

    def expand(self, e, k):
        q = self.vq._up_to_full(e, k, self.context_length) if k != self.K - 1 else e
        return self.vq._apply_quant_resi(q, k)

    # Not vq.idx_to_zhats: it appends one tensor it updates in place, so every entry equals the final level.
    def z_hats(self, codes):
        z_hat, out = 0.0, []
        for k, level_codes in enumerate(self.split_levels(codes)):
            z_hat = z_hat + self.expand(self.vq.embeddings[k](level_codes), k)
            out.append(z_hat)
        return out

    def decode_logits(self, z_hat):
        return self.vqvae.decoder(z_hat)

    def resize(self, z_hat, k):
        return F.interpolate(z_hat.transpose(1, 2), size=self.levels[k], mode="area").transpose(1, 2)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def load_released(task, local_dir=None):
    """Loads T0 for task in {tinystories, xsum}. Returns (tokenizer, config)."""
    ensure_path()
    from models.vqvae import VQVAE
    if local_dir:
        config_path, weights_path = os.path.join(local_dir, "config.yaml"), os.path.join(local_dir, "tokenizer.pt")
    else:
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(HF_CHECKPOINTS, f"{task}/config.yaml")
        weights_path = hf_hub_download(HF_CHECKPOINTS, f"{task}/tokenizer.pt")
    cfg = OmegaConf.load(config_path)
    vqvae = VQVAE(**cfg.tokenizer.vqvae)
    vqvae.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True)["model"], strict=True)
    pad_id, eos_id = {"tinystories": (1, 0), "xsum": (50257, 50256)}[task]
    return NCMTokenizer(vqvae, pad_id, eos_id).eval(), cfg
