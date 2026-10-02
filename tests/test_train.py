import math
import unittest

import torch

from detection.losses import DetectionLoss
from engine.train import train_one_epoch


class TinyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(1, 2, 1))
        self.deltas = torch.nn.Parameter(torch.zeros(1, 2, 4))
        self.register_buffer("anchors", torch.tensor([
            [0., 0., 10., 10.], [0., 0., 9., 9.],
        ]))

    def forward(self, images):
        return self.logits, self.deltas, self.anchors


class TrainProgressTest(unittest.TestCase):
    def test_first_batch_finishes_and_logs_with_tqdm(self):
        for debug in (False, True):
            with self.subTest(debug=debug):
                model = TinyDetector()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                batch = (torch.zeros(1, 3, 16, 16), [{
                    "boxes": torch.tensor([[0., 0., 10., 10.]]),
                    "labels": torch.tensor([1]),
                }])
                with self.assertLogs("engine.train", level="INFO") as captured:
                    metrics = train_one_epoch(
                        model, DetectionLoss(num_classes=1), [batch, batch],
                        optimizer, None, torch.device("cpu"), 0,
                        amp=False, log_interval=1, debug_first_batch=debug,
                    )
                self.assertTrue(math.isfinite(metrics["total_loss"]))
                self.assertFalse(torch.equal(model.logits, torch.zeros_like(model.logits)))
                logs = "\n".join(captured.output)
                self.assertIn("waiting for first batch", logs)
                self.assertIn("First batch received", logs)
                self.assertIn("Epoch 0 [1/2]", logs)
                self.assertIn("Epoch 0 [2/2]", logs)
                if debug:
                    for stage in ("model forward", "loss", "backward", "optimizer update", "complete"):
                        self.assertIn(f"First batch: {'starting ' if stage != 'complete' else ''}{stage}", logs)


if __name__ == "__main__":
    unittest.main()
