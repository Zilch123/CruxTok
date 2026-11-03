import os

import pytest
import torch

from cruxtok.training.runtime import load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE = os.path.join(ROOT, "configs", "smoke", "tiny.yaml")
SMOKE_NCM = os.path.join(ROOT, "configs", "smoke", "tiny_ncm.yaml")


@pytest.fixture
def cfg():
    torch.manual_seed(0)
    return load_config(SMOKE)


@pytest.fixture
def ncm_cfg():
    torch.manual_seed(0)
    return load_config(SMOKE_NCM)
