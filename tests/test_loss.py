import pytest

import torch

from loss import build_loss_aggregator
from utils.others import load_config


def test_loss():
    cfg = load_config("config/test_config.yaml")
    loss_agg = build_loss_aggregator(cfg)

    assert len(loss_agg.components) == 6

    slice_ref = torch.randn((3, 1, 2048))
    slice_est = torch.randn((3, 1, 2048))
    ref = {"slice": slice_ref}
    est = {"slice": slice_est}

    loss = loss_agg.compute(est, ref)

    assert len(loss.individuals) == 6
