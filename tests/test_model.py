import numpy as np
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from cruxtok.data.sources import build_data
from cruxtok.eval.recon import evaluate_reconstruction
from cruxtok.model.cruxtok import CruxTok
from cruxtok.ncm import NCMTokenizer
from cruxtok.training.tokenizer_trainer import TrainModule


def batch(cfg, n=8):
    return torch.randint(2, cfg.model.in_vocab_size, (n, cfg.model.context_length))


def no_autocast():
    return torch.autocast("cpu", enabled=False)


def test_shapes_per_level(cfg):
    m = CruxTok(cfg.model).eval()
    ids = batch(cfg)
    codes = m.encode_ids(ids)
    assert codes.shape == (8, sum(cfg.model.levels))
    for k, (level_codes, l) in enumerate(zip(m.split_levels(codes), cfg.model.levels)):
        assert level_codes.shape == (8, l)
        assert m.code_vectors(codes, k).shape == (8, l, cfg.model.d)
        assert m.expand(m.code_vectors(codes, k), k).shape == (8, cfg.model.context_length, cfg.model.d)
        assert m.resize(m.z_hats(codes)[k], k).shape == (8, l, cfg.model.d)
    for weights, l in zip(m.attention_maps(ids)[:-1], cfg.model.levels):
        assert weights.shape == (8, l, cfg.model.context_length)
        assert torch.allclose(weights.sum(-1), torch.ones(8, l), atol=1e-4)


@pytest.mark.parametrize("fill", ["eos", "pad"])
def test_pooling_ignores_padding(cfg, fill):
    m = CruxTok(cfg.model).eval()
    ids = batch(cfg)
    ids[:, 10:] = getattr(cfg.model, f"{fill}_id")
    weights = m.attention_maps(ids)[0]
    # With an EOS fill, the first EOS is real text and stays visible.
    first_hidden = 11 if fill == "eos" else 10
    assert weights[:, :, first_hidden:].abs().max() < 1e-6, "queries must not attend to padding"
    assert weights[:, :, :first_hidden].sum() > 0


def test_queries_start_on_distinct_windows(cfg):
    """Regression: without a prior, every query pooled the same target and a level held one slot's worth."""
    m = CruxTok(cfg.model).eval()
    weights = m.attention_maps(batch(cfg))[2]
    assert weights.argmax(-1).float().std(1).mean() > 2.0, "slots must attend to different positions"
    H = m.encode_continuous(batch(cfg))
    u = m.run_levels(H).u[2]
    assert u.std(1).mean() > 0.5 * u.pow(2).mean().sqrt(), "slots must carry different vectors"


@pytest.mark.parametrize("ste", ["per_level", "coarse_only"])
def test_gradient_reaches_encoder_through_ste(cfg, ste):
    cfg.model.ste = ste
    m = CruxTok(cfg.model).train()
    ids = batch(cfg)
    out = m(ids, cutoff=m.K - 1)
    F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]), ids.reshape(-1)).backward()
    assert m.wte.weight.grad.abs().sum() > 0
    assert m.encoder_blocks[0].mlp[0].weight.grad.abs().sum() > 0
    if ste == "per_level":
        assert all(p.queries.grad.abs().sum() > 0 for p in m.pools), "every pooler needs reconstruction gradient"
    else:
        assert m.pools[0].queries.grad.abs().sum() > 0


def test_codebooks_are_ema_only(cfg):
    m = CruxTok(cfg.model).train()
    before = [c.embedding.weight.clone() for c in m.codebooks]
    ids = batch(cfg)
    out = m(ids)
    (F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]), ids.reshape(-1)) + out.vq_loss).backward()
    assert all(c.embedding.weight.grad is None for c in m.codebooks)
    assert any(not torch.equal(b, c.embedding.weight) for b, c in zip(before, m.codebooks)), "EMA must update"


def test_no_gradient_chain_across_levels(cfg):
    """Backpropagating from level k must not reach an earlier pooler through the residual."""
    m = CruxTok(cfg.model).train()
    H = m.encode_continuous(batch(cfg))
    levels = m.run_levels(H)
    levels.q[2].pow(2).sum().backward()
    assert m.pools[2].queries.grad is not None
    assert m.pools[0].queries.grad is None and m.pools[1].queries.grad is None


def test_nested_dropout_truncation(cfg):
    m = CruxTok(cfg.model).train()
    ids = batch(cfg)
    torch.manual_seed(0)
    cutoffs = [m(ids).cutoff for _ in range(200)]
    assert set(cutoffs) == set(range(m.K)), "every level must be reachable as a cutoff"
    assert 0.35 < sum(c == m.K - 1 for c in cutoffs) / 200 < 0.75, "the full depth is kept about half the time"
    m.eval()
    assert all(m(ids).cutoff == m.K - 1 for _ in range(5)), "no truncation at eval"


def test_truncated_logits_ignore_finer_levels(cfg):
    """A cutoff at level k must decode exactly the sum of levels 1..k."""
    m = CruxTok(cfg.model).eval()
    ids = batch(cfg)
    codes = m.encode_ids(ids)
    with torch.no_grad():
        for k in range(m.K):
            direct = m(ids, cutoff=k).logits
            assert torch.allclose(direct, m.decode_logits(m.z_hats(codes)[k]), atol=1e-4), f"level {k}"


def test_decoder_works_on_any_prefix(cfg):
    m = CruxTok(cfg.model).eval()
    ids = batch(cfg)
    z_hats = m.z_hats(m.encode_ids(ids))
    assert len(z_hats) == m.K
    for z_hat in z_hats:
        assert m.decode_logits(z_hat).shape == (8, cfg.model.context_length, cfg.model.in_vocab_size)
    assert not torch.equal(z_hats[0], z_hats[-1])


def test_final_level_identity_flag(cfg):
    assert len(CruxTok(cfg.model).pools) == len(cfg.model.levels) - 1
    cfg.model.final_level_identity = False
    m = CruxTok(cfg.model).eval()
    assert len(m.pools) == len(cfg.model.levels)
    assert m.encode_ids(batch(cfg)).shape[1] == sum(cfg.model.levels)


def test_fp16_path_stays_finite(cfg):
    m = TrainModule(cfg).train()
    autocast = lambda: torch.autocast("cpu", dtype=torch.float16)
    loss, metrics = m.loss(batch(cfg), autocast)
    assert np.isfinite(metrics["loss"])
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    assert m.tokenizer.codebooks[0].embedding.weight.dtype == torch.float32


def test_ncm_wrapper_interface_and_cumulative_levels(ncm_cfg):
    m = NCMTokenizer.from_model_config(ncm_cfg.model).eval()
    ids = batch(ncm_cfg)
    codes = m.encode_ids(ids)
    assert codes.shape == (8, sum(ncm_cfg.model.levels))
    z_hats = m.z_hats(codes)
    assert not torch.equal(z_hats[0], z_hats[-1]), "levels must be distinct, unlike vq.idx_to_zhats"
    assert torch.allclose(z_hats[-1], m.vq.idx_to_z_q(codes), atol=1e-4), "must agree with NCM's own full reconstruction"
    r = evaluate_reconstruction(m, ids.numpy(), 4)
    assert len(r["accuracy"]) == m.K


def test_ncm_trainer_loss_matches_eq9_as_coded(ncm_cfg):
    """T0-cap through TrainModule equals NCM's own forward plus cross-entropy."""
    ncm_cfg.model.nested_dropout.prob = 0.0
    m = TrainModule(ncm_cfg).eval()
    ids = batch(ncm_cfg)
    with torch.no_grad():
        loss, metrics = m.loss(ids, no_autocast)
        logits, vq_loss, *_ = m.tokenizer.vqvae(ids)
        want = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids.reshape(-1)) + vq_loss
    assert abs(loss.item() - want.item()) < 1e-4


def test_public_api_round_trip(cfg, tmp_path):
    cfg.model.text_tokenizer = "bytes"
    cfg.model.in_vocab_size = 258
    m = CruxTok(cfg.model).eval()
    levels = m.encode("hello there")
    assert [c.shape[1] for c in levels] == list(cfg.model.levels)
    assert isinstance(m.decode(levels), str) and isinstance(m.decode(levels, upto_level=1), str)
    m.save_pretrained(str(tmp_path))
    loaded = CruxTok.from_pretrained(str(tmp_path))
    assert all(torch.equal(a, b) for a, b in zip(levels, loaded.encode("hello there")))


def test_attention_report(cfg):
    from cruxtok.eval.attention import attention_report
    m = CruxTok(cfg.model).eval()
    rows = attention_report(m, batch(cfg), cfg.model.prior_sharpness)
    assert len(rows) == len(cfg.model.levels) - 1
    assert rows[0]["prior_vs_start"] < 1e-5, "an untrained prior has not moved"
    assert rows[1]["top_token_spread_between_slots"] > 0


def test_qk_norm_bounds_attention_logits():
    from cruxtok.model.layers import Attention
    torch.manual_seed(0)
    x = torch.randn(2, 16, 64)
    for qk_norm in (False, True):
        attn = Attention(64, 1, qk_norm=qk_norm)  # one head: returned weights are a mean over heads
        with torch.no_grad():
            attn.q.weight.mul_(1000)
            attn.k.weight.mul_(1000)
        probs = attn(x, x, x, need_weights=True)[1]
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1).mean()
        # Inflated weights saturate plain attention; with qk_norm the logits stay at most sqrt(head_dim).
        assert (entropy > 0.5) == qk_norm, (qk_norm, entropy)


def test_semantic_eval_shapes_and_bootstrap(cfg):
    from cruxtok.eval.semantic import code_similarities, paired_bootstrap
    cfg.model.text_tokenizer = "bytes"
    m = CruxTok(cfg.model).eval()
    s1, s2 = ["a cat sat", "rain fell", "dogs bark"], ["a cat sat", "sun shone", "cats meow"]
    sims = code_similarities(m, cfg, s1, s2, max_k=3)
    assert sims.shape == (3, 3)
    assert np.allclose(sims[0], 1.0, atol=1e-5)  # identical text, identical codes
    labels = np.array([1.0, 0.0, 1.0, 0.0])
    same = paired_bootstrap("paws", labels, np.array([.9, .1, .8, .2]), np.array([.9, .1, .8, .2]), n=50)
    assert same["diff"] == 0 and same["lo"] == 0 and same["hi"] == 0
