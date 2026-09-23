import unittest

import torch
import torch.nn as nn

from detection.fastvit_detector import FastViTDetector
from engine.evaluate import _restore_boxes_to_original


class _PredictHarness(nn.Module):
    predict = FastViTDetector.predict

    def forward(self, images):
        # The second anchor has the highest score but lies in right/bottom
        # padding for a 50x50 valid image inside a 128x128 batch tensor.
        cls_logits = torch.tensor([[[-2.0], [10.0]]], device=images.device)
        box_deltas = torch.zeros((1, 2, 4), device=images.device)
        anchors = torch.tensor(
            [[5.0, 5.0, 15.0, 15.0], [85.0, 85.0, 95.0, 95.0]],
            device=images.device,
        )
        return cls_logits, box_deltas, anchors


class DetectorPostprocessTest(unittest.TestCase):
    def test_padding_anchor_is_removed_before_topk(self):
        model = _PredictHarness()
        predictions = model.predict(
            torch.zeros(1, 3, 128, 128),
            score_thresh=0.01,
            pre_nms_topk=1,
            image_sizes=[(50, 50)],
        )
        self.assertEqual(len(predictions[0]["boxes"]), 1)
        self.assertTrue((predictions[0]["boxes"] <= 50).all())

    def test_boxes_restore_to_original_non_square_image(self):
        restored = _restore_boxes_to_original(
            torch.tensor([[16.0, 8.0, 48.0, 24.0]]),
            resized_size=(32, 64),
            original_size=(20, 40),
        )
        expected = torch.tensor([[10.0, 5.0, 30.0, 15.0]])
        self.assertTrue(torch.allclose(restored, expected))


if __name__ == "__main__":
    unittest.main()
