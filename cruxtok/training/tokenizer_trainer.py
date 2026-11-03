import argparse
import json
import math
import os
import signal
import subprocess
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from cruxtok.data.sources import build_data
from cruxtok.eval.recon import evaluate_reconstruction
from cruxtok.model.cruxtok import CruxTok
from cruxtok.ncm import NCMTokenizer
from cruxtok.training import runtime


def build_tokenizer(m):
    if m.arch == "cruxtok":
        return CruxTok(m)
    if m.arch == "ncm":
        return NCMTokenizer.from_model_config(m)
    raise ValueError(f"unknown arch {m.arch}")


class TrainModule(nn.Module):
    """The tokenizer and its loss. All loss terms are computed in full precision."""

    def __init__(self, cfg):
        super().__init__()
        self.tokenizer = build_tokenizer(cfg.model)

    def loss(self, ids, autocast):
        with autocast():
            out = self.tokenizer(ids)
        with torch.autocast(ids.device.type, enabled=False):
            logits = out.logits.float()
            rec = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids.reshape(-1))
            total = rec + out.vq_loss
        metrics = {"rec": rec.item(), "vq": out.vq_loss.item(), "cutoff": out.cutoff}
        metrics["loss"] = total.item()
        metrics["perplexity"] = out.perplexity
        return total, metrics


def run(cfg, resume_from="", max_hours=11.5, benchmark=0):
    t, rt = cfg.train, cfg.runtime
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_rng = runtime.set_seed(cfg.seed)
    windows = build_data(cfg, cfg.seed)
    model = TrainModule(cfg).to(device)
    print(f"[train] {cfg.run_name}: tokenizer {model.tokenizer.num_params() / 1e6:.2f}M params", flush=True)
    if rt.compile and cfg.model.arch == "cruxtok":
        for block in [*model.tokenizer.encoder_blocks, *model.tokenizer.decoder_blocks]:
            block.compile()

    opt = torch.optim.AdamW(runtime.weight_decay_groups(model, t.weight_decay), lr=t.lr, betas=tuple(t.betas))
    scaler = torch.amp.GradScaler(device.type, enabled=rt.precision == "fp16" and device.type == "cuda")
    dtype = runtime.amp_dtype(rt.precision)
    autocast = lambda: torch.autocast(device.type, dtype=dtype, enabled=dtype is not None)

    run_dir = os.path.join(rt.output_root, cfg.run_name)
    step, wandb_id, resume = 0, None, None
    if not benchmark:
        os.makedirs(run_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(run_dir, "config.yaml"))
        # A checkpoint in this run's own dir is newer than any attached input.
        resume = runtime.latest_checkpoint(run_dir) or (runtime.latest_checkpoint(resume_from) if resume_from else None)
        if resume_from and not resume:
            raise SystemExit(f"no checkpoint found at {resume_from}")
    if resume:
        step, wandb_id = runtime.load_checkpoint(resume, model, opt, scaler, data_rng, device)
        print(f"[train] resumed {resume} at step {step}", flush=True)

    wandb_run = None
    if rt.wandb and os.environ.get("WANDB_API_KEY") and not benchmark:
        import wandb
        wandb_id = wandb_id or wandb.util.generate_id()
        wandb_run = wandb.init(project=rt.wandb_project, name=cfg.run_name, id=wandb_id, resume="allow",
                               config=OmegaConf.to_container(cfg, resolve=True))

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    heldout = None if benchmark else windows.heldout(rt.eval_examples)
    total_steps = benchmark or t.steps
    start, timer, timed, nonfinite = time.time(), None, 0, 0
    model.train()
    print(f"[train] device={device} precision={rt.precision} steps={total_steps} batch={t.micro_batch}x{t.grad_accum}", flush=True)

    while step < total_steps:
        tick = time.time()
        for group in opt.param_groups:
            group["lr"] = runtime.lr_at(step, t)
        opt.zero_grad(set_to_none=True)
        for _ in range(t.grad_accum):
            ids = torch.as_tensor(windows.sample(t.micro_batch, data_rng), device=device)
            loss, metrics = model.loss(ids, autocast)
            if not math.isfinite(metrics["loss"]):
                raise SystemExit(f"non-finite loss at step {step}: {metrics}")
            scaler.scale(loss / t.grad_accum).backward()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
        scaler.step(opt)
        scaler.update()
        step += 1
        # The fp16 scaler skips non-finite updates; a long run of them means training has silently stopped.
        nonfinite = 0 if math.isfinite(float(grad_norm)) else nonfinite + 1
        if t.max_nonfinite_steps and nonfinite >= t.max_nonfinite_steps:
            raise SystemExit(f"{nonfinite} updates in a row had non-finite gradients, ending at step {step}")
        step_time = time.time() - tick

        if benchmark:
            # The first updates include allocation and warm-up.
            if step == min(5, total_steps - 1):
                timer = time.time()
            elif timer is not None:
                timed += 1
            continue

        if step % rt.log_every == 0 or step == 1:
            row = {"step": step, "lr": opt.param_groups[0]["lr"], "grad_norm": float(grad_norm), "step_time": step_time,
                   "elapsed_h": (time.time() - start) / 3600, **metrics}
            runtime.append_jsonl(os.path.join(run_dir, "metrics.jsonl"), row)
            print(f"step {step} loss {metrics['loss']:.4f} rec {metrics['rec']:.4f} vq {metrics['vq']:.4f} "
                  f"grad {float(grad_norm):.2f} {step_time:.2f}s/step", flush=True)
            if wandb_run:
                flat = {k: v for k, v in row.items() if k != "perplexity"}
                flat.update({f"perplexity/level_{k + 1}": v for k, v in enumerate(metrics["perplexity"])})
                wandb_run.log(flat, step=step)

        hit_limit = stop["flag"] or (time.time() - start) / 3600 >= max_hours
        if step % rt.ckpt_every == 0 or hit_limit or step == total_steps:
            runtime.save_checkpoint(run_dir, step, model, opt, scaler, data_rng, wandb_id, rt.keep_ckpts)
        if hit_limit:
            print(f"[train] stopping cleanly at step {step}; resume to continue", flush=True)
            break
        if (rt.eval_every and step % rt.eval_every == 0) or step == total_steps:
            run_eval(model, heldout, cfg, step, run_dir, wandb_run)

    if benchmark:
        spu = (time.time() - timer) / max(timed, 1)
        peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        print(json.dumps({"seconds_per_update": spu, "peak_gpu_gb": peak, "timed_updates": timed}), flush=True)
        return None

    finished = step == total_steps
    if finished:
        torch.save({"model": model.tokenizer.state_dict(), "config": OmegaConf.to_container(cfg, resolve=True)},
                   os.path.join(run_dir, "tokenizer.pt"))
        print(f"[train] finished at step {step}", flush=True)
    if wandb_run:
        wandb_run.finish()
    return run_dir if finished else None


def run_eval(model, heldout, cfg, step, run_dir, wandb_run):
    tick = time.time()
    r = evaluate_reconstruction(model.tokenizer, heldout, cfg.runtime.eval_batch_size)
    row = {"step": step, **r}
    runtime.append_jsonl(os.path.join(run_dir, "eval.jsonl"), row)
    fmt = lambda xs: " ".join(f"{x:.1f}" for x in xs)
    print(f"[eval] step {step} acc per level: {fmt(r['accuracy'])} | text only: {fmt(r['accuracy_text'])}", flush=True)
    print(f"[eval] used% {fmt(r['utilization'])} | perplexity {fmt(r['perplexity'])} | {time.time() - tick:.0f}s", flush=True)
    if wandb_run:
        for name, values in r.items():
            wandb_run.log({f"eval/{name}_level_{k + 1}": v for k, v in enumerate(values)}, step=step)


def main():
    ap = argparse.ArgumentParser(description="train a tokenizer; overrides use dotted keys, e.g. train.steps=200")
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume_from", default="", help="checkpoint file or directory; default: latest in the run dir")
    ap.add_argument("--max_hours", type=float, default=11.5, help="save and exit cleanly after this many hours")
    ap.add_argument("--benchmark", type=int, default=0, help="time this many updates, print seconds per update, exit")
    ap.add_argument("--auto_stop", action="store_true", help="stop the RunPod pod when this run ends")
    a, overrides = ap.parse_known_args()
    try:
        run(runtime.load_config(a.config, overrides), a.resume_from, a.max_hours, a.benchmark)
    finally:
        if a.auto_stop and os.environ.get("RUNPOD_POD_ID"):
            subprocess.run(["runpodctl", "stop", "pod", os.environ["RUNPOD_POD_ID"]], check=False)


if __name__ == "__main__":
    main()
