import unittest

import torch

from loss import LossAggregatorFactory
from utils.others import load_config


class TestLoss(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config("config/test_config.yaml")

    def test_loss_aggregation(self):
        loss_agg = LossAggregatorFactory.build_loss_aggregator(self.cfg["loss"])
        self.assertEqual(len(loss_agg.loss_components), 6)

        slice_ref = torch.randn((3, 1, 2048))
        slice_est = torch.randn((3, 1, 2048))
        ref = {"slice": slice_ref}
        est = {"slice": slice_est}

        loss = loss_agg(est, ref)
        self.assertEqual(len(loss), 7)


if __name__ == "__main__":
    unittest.main()
