import unittest

import torch

from detection.losses import DetectionLoss


class DetectionLossTest(unittest.TestCase):
    def test_overlapping_ground_truths_receive_distinct_anchors(self):
        criterion = DetectionLoss(num_classes=10, reg_max=16)
        cls_predictions = torch.zeros((1, 2, 10), requires_grad=True)
        box_predictions = torch.zeros((1, 2, 4), requires_grad=True)
        anchors = torch.tensor(
            [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 9.0, 9.0]]
        )
        targets = [
            {
                "boxes": torch.tensor(
                    [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]]
                ),
                "labels": torch.tensor([1, 2]),
            }
        ]

        result = criterion(cls_predictions, box_predictions, anchors, targets)
        total = result["cls_loss"] + result["reg_loss"]
        total.backward()

        self.assertEqual(result["num_pos"], 2)
        self.assertTrue(torch.isfinite(total))
        self.assertIsNotNone(cls_predictions.grad)


if __name__ == "__main__":
    unittest.main()
