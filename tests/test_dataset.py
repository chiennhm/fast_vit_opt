import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from bdd100k_dataset import BDD100KDetectionDataset, bdd100k_collate


class BDD100KDatasetTest(unittest.TestCase):
    def test_box_only_target_and_empty_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            Image.new("RGB", (40, 20), "white").save(images / "one.jpg")
            Image.new("RGB", (40, 20), "black").save(images / "empty.jpg")
            annotations = {
                "images": [
                    {"id": 1, "file_name": "one.jpg", "width": 40, "height": 20},
                    {"id": 2, "file_name": "empty.jpg", "width": 40, "height": 20},
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 3, "bbox": [5, 2, 10, 8]}
                ],
                "categories": [
                    {"id": index + 1, "name": name}
                    for index, name in enumerate(
                        [
                            "bike", "bus", "car", "motor", "person", "rider",
                            "traffic light", "traffic sign", "train", "truck",
                        ]
                    )
                ],
            }
            annotation_path = root / "labels.json"
            annotation_path.write_text(json.dumps(annotations), encoding="utf-8")

            dataset = BDD100KDetectionDataset(
                images, annotation_path, img_size=32, augment=False
            )
            image, target = dataset[0]
            empty_image, empty_target = dataset[1]

            self.assertEqual(tuple(image.shape), (3, 32, 64))
            self.assertEqual(tuple(empty_image.shape), (3, 32, 64))
            self.assertEqual(target["labels"].tolist(), [3])
            self.assertEqual(target["image_id"].item(), 1)
            self.assertEqual(target["orig_size"].tolist(), [20, 40])
            self.assertEqual(target["size"].tolist(), [32, 64])
            self.assertEqual(target["area"].tolist(), [80.0])
            self.assertNotIn("masks", target)
            self.assertEqual(tuple(empty_target["boxes"].shape), (0, 4))

            batch, targets = bdd100k_collate([(image, target), (empty_image, empty_target)])
            self.assertEqual(tuple(batch.shape), (2, 3, 32, 64))
            self.assertEqual(len(targets), 2)


if __name__ == "__main__":
    unittest.main()
