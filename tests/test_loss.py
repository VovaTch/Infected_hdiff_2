import pytest

import torch

from loss import build_loss_aggregator
from utils.others import load_config


def test_loss():
    cfg = load_config("config/test_config.yaml")
    del cfg["loss"]["melspec_rec_loss_1"], cfg["loss"]["rec_noise_loss"]
    loss_agg = build_loss_aggregator(cfg)

    assert len(loss_agg.components) == 6

    slice_ref = torch.randn((3, 1, 2048))
    slice_est = torch.randn((3, 1, 2048))
    ref = {"slice": slice_ref}
    est = {"slice": slice_est}

    loss = loss_agg.compute(est, ref)

    assert len(loss.individuals) == 6


def test_diff_rec_losses():
    cfg = load_config("config/test_config.yaml")
    cfg_new = {}
    for key in cfg["loss"]:
        if key in ["aggregator_type", "melspec_rec_loss_1", "rec_noise_loss"]:
            cfg_new[key] = cfg["loss"][key]
    loss_agg = build_loss_aggregator({"loss": cfg_new})

    assert len(loss_agg.components) == 2

    slice_ref = torch.randn((3, 1, 2048))
    slice_est = torch.randn((3, 1, 2048))
    noisy_slice = torch.randn_like(slice_ref)
    noise_pred = torch.randn_like(slice_ref)
    ref = {"slice": slice_ref, "noisy_slice": noisy_slice, "noise_scale": 0.10}
    est = {"slice": slice_est, "noise_pred": noise_pred}

    loss = loss_agg.compute(est, ref)

    assert len(loss.individuals) == 2
