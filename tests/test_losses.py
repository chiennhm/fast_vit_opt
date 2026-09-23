import unittest

import torch

from detection.losses import DetectionLoss, QualityFocalLoss


class QualityFocalLossTest(unittest.TestCase):
    def test_positive_target_uses_continuous_quality(self):
        criterion = QualityFocalLoss(beta=2.0)
        labels = torch.tensor([1])
        quality = torch.tensor([0.7])
        calibrated = torch.logit(torch.tensor([[0.7]]))
        overconfident = torch.logit(torch.tensor([[0.95]]))

        calibrated_loss = criterion(calibrated, labels, quality)
        overconfident_loss = criterion(overconfident, labels, quality)

        self.assertLess(calibrated_loss.item(), overconfident_loss.item())

    def test_quality_target_is_detached(self):
        criterion = QualityFocalLoss(beta=2.0)
        logits = torch.zeros((1, 1), requires_grad=True)
        quality = torch.tensor([0.6], requires_grad=True)
        loss = criterion(logits, torch.tensor([1]), quality)
        loss.backward()

        self.assertIsNotNone(logits.grad)
        self.assertIsNone(quality.grad)


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

    def test_qfl_classification_does_not_backpropagate_through_iou_target(self):
        criterion = DetectionLoss(num_classes=1, classification_loss="qfl")
        cls_predictions = torch.zeros((1, 1, 1), requires_grad=True)
        box_predictions = torch.zeros((1, 1, 4), requires_grad=True)
        anchors = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        targets = [
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                "labels": torch.tensor([1]),
            }
        ]

        result = criterion(cls_predictions, box_predictions, anchors, targets)
        result["cls_loss"].backward()

        self.assertIsNotNone(cls_predictions.grad)
        self.assertIsNone(box_predictions.grad)


if __name__ == "__main__":
    unittest.main()
