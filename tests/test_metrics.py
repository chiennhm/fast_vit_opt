import unittest

import numpy as np

from detection import metrics


@unittest.skipUnless(metrics.HAS_PYCOCOTOOLS, "pycocotools is not installed")
class DetectionMetricsTest(unittest.TestCase):
    def test_perfect_prediction_has_perfect_map(self):
        prediction = {
            "boxes": np.array([[1, 1, 11, 11]], dtype=np.float32),
            "labels": np.array([3], dtype=np.int64),
            "scores": np.array([0.99], dtype=np.float32),
        }
        target = {
            "boxes": np.array([[1, 1, 11, 11]], dtype=np.float32),
            "labels": np.array([3], dtype=np.int64),
            "iscrowd": np.array([0], dtype=np.int64),
        }
        result = metrics.evaluate_coco(
            [prediction],
            [target],
            num_classes=10,
            iou_threshold=[0.5, 0.75],
            class_names=[str(index) for index in range(10)],
        )
        self.assertAlmostEqual(result["mAP"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
