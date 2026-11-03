import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from cruxtok.model.expansion import Expansion
from cruxtok.model.layers import Block, padding_positions
from cruxtok.model.pooling import QueryPool
from cruxtok.model.vq import LevelCodebook


@dataclass
class LevelOutput:
    e: list  # per-level vectors carrying gradient to the pooler where STE is on
    u: list  # pooled residuals
    e_raw: list  # selected code vectors
    codes: list
    q: list  # R_k(e_k), each [B, L, d]
    weights: list  # pooling attention, [B, l_k, L]; entries are None unless requested


@dataclass
class TokenizerOutput:
    logits: torch.Tensor
    vq_loss: torch.Tensor
    codes: torch.Tensor
    cutoff: int
    perplexity: list


class CruxTok(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, L = cfg.d, cfg.context_length
        self.levels = list(cfg.levels)
        self.K = len(self.levels)
        self.vocab_sizes = [cfg.codebook_size] * self.K
        self.embed_dim, self.context_length, self.in_vocab_size = d, L, cfg.in_vocab_size
        self.pad_id, self.eos_id = cfg.pad_id, cfg.eos_id
        self.ste, self.m = cfg.ste, cfg.m
        assert self.levels[-1] == L, f"the last level must have length {L}"
        assert self.ste in ("per_level", "coarse_only"), self.ste
        self.identity_last = bool(cfg.final_level_identity)
        self.commitment = cfg.vq.commitment
        self.fine_drop_prob, self.fine_drop_min_keep = cfg.nested_dropout.prob, cfg.nested_dropout.min_keep

        self.wte = nn.Embedding(cfg.in_vocab_size, d)
        self.wpe = nn.Embedding(L, d)
        self.drop = nn.Dropout(cfg.dropout)
        self.encoder_blocks = nn.ModuleList(Block(d, cfg.n_heads, cfg.dropout) for _ in range(cfg.encoder_blocks))
        # Not affine: ||z - e||^2 is not scale-invariant, so a learned gain could inflate without bound.
        self.pre_quant_norm = nn.LayerNorm(d, elementwise_affine=False)

        n_learned = self.K - 1 if self.identity_last else self.K
        # Absent in checkpoints saved before the flag existed.
        qk_norm = cfg.get("qk_norm", False)
        self.pools = nn.ModuleList(
            QueryPool(l, L, d, cfg.n_heads, cfg.dropout, cfg.pool_self_attn, cfg.attn_prior, cfg.query_init_std, cfg.prior_sharpness, qk_norm)
            for l in self.levels[:n_learned])
        self.expansions = nn.ModuleList(
            Expansion(l, L, d, cfg.n_heads, cfg.dropout, cfg.attn_prior, cfg.query_init_std, cfg.prior_sharpness, qk_norm) for l in self.levels[:n_learned])
        self.codebooks = nn.ModuleList(
            LevelCodebook(cfg.codebook_size, d, cfg.vq.decay, cfg.vq.epsilon, cfg.vq.restart_threshold) for _ in self.levels)

        self.dec_wpe = nn.Embedding(L, d)
        self.decoder_blocks = nn.ModuleList(Block(d, cfg.n_heads, cfg.dropout) for _ in range(cfg.decoder_blocks))
        self.ln_out = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.in_vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding) and module.weight.requires_grad:
            nn.init.normal_(module.weight, std=0.02)

    @property
    def embeddings(self):
        return [c.embedding for c in self.codebooks]

    def padding_mask(self, ids):
        mask = padding_positions(ids, self.pad_id, self.eos_id)
        mask[:, 0] = False  # an all-masked row has no attention target
        return mask

    def encode_continuous(self, ids, key_padding_mask=None):
        x = self.drop(self.wte(ids) + self.wpe(torch.arange(ids.shape[1], device=ids.device)))
        for block in self.encoder_blocks:
            x = block(x, key_padding_mask)
        return self.pre_quant_norm(x)

    def _uses_ste(self, k):
        return self.ste == "per_level" or k < self.m

    def _is_identity(self, k):
        return self.identity_last and k == self.K - 1

    def _pool(self, residual, k, mask, need_weights):
        return (residual, None) if self._is_identity(k) else self.pools[k](residual, mask, need_weights)

    def expand(self, e, k):
        return e if self._is_identity(k) else self.expansions[k](e)

    # r^(k) stops at z_hat so gradients never chain across levels, which is what makes per-level STE stable.
    def run_levels(self, H, mask=None, upto=None, update_ema=False, need_weights=False):
        z_hat = torch.zeros_like(H)
        out = LevelOutput([], [], [], [], [], [])
        for k in range(upto or self.K):
            u, weights = self._pool(H - z_hat.detach(), k, mask, need_weights)
            e_raw, idx = self.codebooks[k](u, update=update_ema)
            e = u + (e_raw - u).detach() if self._uses_ste(k) else e_raw
            q = self.expand(e, k)
            z_hat = z_hat + q
            for store, value in zip((out.e, out.u, out.e_raw, out.codes, out.q, out.weights), (e, u, e_raw, idx, q, weights)):
                store.append(value)
        return out

    def _sample_cutoff(self):
        last = self.K - 1
        if not self.training or float(torch.rand(())) >= self.fine_drop_prob:
            return last
        low, high = max(self.fine_drop_min_keep - 1, 0), last - 1
        return int(torch.randint(low, high + 1, ())) if high >= low else last

    def _decoder_input(self, H, q, cutoff):
        if self.ste == "per_level" or cutoff < self.m:
            return sum(q[:cutoff + 1])
        # coarse_only: below level m, gradients skip to the encoder in one step as in NCM.
        coarse = sum(q[:self.m])
        target = H - coarse.detach()
        fine = sum(q[self.m:cutoff + 1])
        return coarse + target + (fine - target).detach()

    def decode_logits(self, z_hat):
        x = z_hat + self.dec_wpe(torch.arange(z_hat.shape[1], device=z_hat.device))
        for block in self.decoder_blocks:
            x = block(x)
        return self.head(self.ln_out(x))

    def forward(self, ids, cutoff=None):
        mask = self.padding_mask(ids)
        H = self.encode_continuous(ids, mask)
        levels = self.run_levels(H, mask, update_ema=self.training)
        cutoff = self._sample_cutoff() if cutoff is None else cutoff
        logits = self.decode_logits(self._decoder_input(H, levels.q, cutoff))
        return TokenizerOutput(logits, self._vq_loss(H, levels), torch.cat(levels.codes, 1), cutoff,
                               [c.perplexity() for c in self.codebooks])

    def _vq_loss(self, H, levels):
        with torch.autocast(H.device.type, enabled=False):
            H = H.float()
            z_hat, loss = 0.0, 0.0
            for u, e_raw, q in zip(levels.u, levels.e_raw, levels.q):
                z_hat = z_hat + q.float()
                loss = loss + self.commitment * F.mse_loss(u.float(), e_raw)
                loss = loss + self.commitment * F.mse_loss(H, z_hat.detach()) + F.mse_loss(z_hat, H.detach())
            return loss / self.K

    @torch.no_grad()
    def encode_ids(self, ids, attention_mask=None):
        mask = self.padding_mask(ids) if attention_mask is None else ~attention_mask.bool()
        return torch.cat(self.run_levels(self.encode_continuous(ids, mask), mask).codes, 1)

    @torch.no_grad()
    def attention_maps(self, ids):
        mask = self.padding_mask(ids)
        return self.run_levels(self.encode_continuous(ids, mask), mask, need_weights=True).weights

    def split_levels(self, codes):
        return list(codes.split(self.levels, dim=1))

    def code_vectors(self, codes, k):
        return self.codebooks[k].embedding(self.split_levels(codes)[k])

    def z_hats(self, codes):
        z_hat, out = 0.0, []
        for k, level_codes in enumerate(self.split_levels(codes)):
            z_hat = z_hat + self.expand(self.codebooks[k].embedding(level_codes), k)
            out.append(z_hat)
        return out

    def resize(self, z_hat, k):
        return F.interpolate(z_hat.transpose(1, 2), size=self.levels[k], mode="area").transpose(1, 2)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_pretrained(cls, path, filename="tokenizer.pt", device="cpu"):
        """Loads a run directory or a Hugging Face repo id holding tokenizer.pt."""
        from omegaconf import OmegaConf
        local = os.path.join(path, filename)
        if not os.path.isfile(local):
            from huggingface_hub import hf_hub_download
            local = hf_hub_download(path, filename)
        state = torch.load(local, map_location=device, weights_only=True)
        model = cls(OmegaConf.create(state["config"]["model"]))
        model.load_state_dict(state["model"])
        model.config = state["config"]
        return model.to(device).eval()

    def _text(self):
        if not hasattr(self, "_text_tokenizer"):
            from cruxtok.data.text import load_text_tokenizer
            self._text_tokenizer = load_text_tokenizer(self.cfg.text_tokenizer)
        return self._text_tokenizer

    @torch.no_grad()
    def encode(self, text):
        """Text to a list of code tensors, one per level, coarse to fine."""
        from cruxtok.data.text import fit_window
        device = self.wte.weight.device
        ids = torch.tensor([fit_window(self._text().encode(text), self.context_length, self.eos_id)], device=device)
        return self.split_levels(self.encode_ids(ids))

    @torch.no_grad()
    def decode(self, codes, upto_level=None):
        """Codes to text, using levels 1..upto_level (default all). Stops at the first EOS."""
        flat = torch.cat(list(codes), 1) if isinstance(codes, (list, tuple)) else codes
        k = self.K if upto_level is None else upto_level
        tokens = self.decode_logits(self.z_hats(flat)[k - 1]).argmax(-1)[0].tolist()
        if self.eos_id in tokens:
            tokens = tokens[:tokens.index(self.eos_id)]
        return self._text().decode(tokens)

    def save_pretrained(self, path, config=None):
        from omegaconf import OmegaConf
        os.makedirs(path, exist_ok=True)
        config = config or getattr(self, "config", None) or {"model": OmegaConf.to_container(self.cfg, resolve=True)}
        torch.save({"model": self.state_dict(), "config": config}, os.path.join(path, "tokenizer.pt"))

    def push_to_hub(self, repo_id, private=True, token=None):
        import tempfile
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo_id, private=private, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            self.save_pretrained(tmp)
            api.upload_folder(repo_id=repo_id, folder_path=tmp)
