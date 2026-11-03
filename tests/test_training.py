import hashlib
import json
import os
import subprocess
import sys

import pytest
import torch

from cruxtok.training.runtime import lr_at
from conftest import ROOT, SMOKE, SMOKE_NCM

CPU = dict(os.environ, CUDA_VISIBLE_DEVICES="")


def _train(out, run, steps=10, config=SMOKE, extra=()):
    cmd = [sys.executable, "-m", "cruxtok.cli.train_tokenizer", "--config", config, f"runtime.output_root={out}", f"run_name={run}",
           f"train.steps={steps}", "train.micro_batch=8", "runtime.ckpt_every=5", "runtime.eval_every=0",
           "runtime.eval_examples=8", "runtime.log_every=5", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, env=CPU, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    return r.stdout


def _final_hash(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(sd[k].numpy().tobytes())
    return h.hexdigest()


def test_resume_is_exact(tmp_path):
    out = str(tmp_path)
    _train(out, "straight")
    _train(out, "cut")
    # Simulate a crash after step 5: drop everything newer, then resume.
    cut = os.path.join(out, "cut")
    assert sorted(f for f in os.listdir(cut) if f.startswith("ckpt_")) == ["ckpt_0000005.pt", "ckpt_0000010.pt"]
    os.remove(os.path.join(cut, "ckpt_0000010.pt"))
    os.remove(os.path.join(cut, "tokenizer.pt"))
    log = _train(out, "cut")
    assert "at step 5" in log, log[-500:]
    assert _final_hash(os.path.join(out, "straight", "tokenizer.pt")) == _final_hash(os.path.join(cut, "tokenizer.pt"))


def test_resume_is_exact_ncm(tmp_path):
    out = str(tmp_path)
    _train(out, "straight", config=SMOKE_NCM)
    _train(out, "cut", config=SMOKE_NCM)
    cut = os.path.join(out, "cut")
    os.remove(os.path.join(cut, "ckpt_0000010.pt"))
    os.remove(os.path.join(cut, "tokenizer.pt"))
    _train(out, "cut", config=SMOKE_NCM)
    assert _final_hash(os.path.join(out, "straight", "tokenizer.pt")) == _final_hash(os.path.join(cut, "tokenizer.pt"))


def test_max_hours_exits_cleanly(tmp_path):
    out = str(tmp_path)
    r = subprocess.run([sys.executable, "-m", "cruxtok.cli.train_tokenizer", "--config", SMOKE, "--max_hours", "0",
                        f"runtime.output_root={out}", "run_name=limit", "train.micro_batch=4"],
                       capture_output=True, text=True, env=CPU, cwd=ROOT)
    assert r.returncode == 0 and "stopping cleanly at step 1" in r.stdout, r.stdout[-500:] + r.stderr[-500:]
    assert "ckpt_0000001.pt" in os.listdir(os.path.join(out, "limit"))


def test_run_writes_resolved_config_and_metrics(tmp_path):
    out = str(tmp_path)
    _train(out, "logged")
    run_dir = os.path.join(out, "logged")
    assert os.path.isfile(os.path.join(run_dir, "config.yaml"))
    with open(os.path.join(run_dir, "metrics.jsonl")) as f:
        rows = [json.loads(line) for line in f]
    assert {"rec", "vq", "perplexity", "step_time"} <= set(rows[-1])


def test_benchmark_reports_speed(tmp_path):
    r = subprocess.run([sys.executable, "-m", "cruxtok.cli.train_tokenizer", "--config", SMOKE, "--benchmark", "8",
                        f"runtime.output_root={tmp_path}", "train.micro_batch=4"],
                       capture_output=True, text=True, env=CPU, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-500:]
    assert json.loads(r.stdout.strip().splitlines()[-1])["seconds_per_update"] > 0


def test_lr_schedule_endpoints():
    from omegaconf import OmegaConf
    t = OmegaConf.create(dict(steps=1000, warmup=100, lr=5e-4, lr_floor=5e-5))
    assert lr_at(0, t) == pytest.approx(5e-6)
    assert lr_at(99, t) == pytest.approx(5e-4)
    assert lr_at(1000, t) == pytest.approx(5e-5)
