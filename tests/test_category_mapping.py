import unittest

from bdd100k_dataset import BDD100K_CLASSES as DATASET_CLASSES
from detection.visualize import BDD100K_CLASSES as VISUALIZATION_CLASSES
from tools.compute_map import BDD100K_CLASSES as ANALYSIS_CLASSES
from tools.split_by_weather import BDD100K_CATEGORIES as WEATHER_CATEGORIES
from tools.visualize_failures import BDD100K_CLASSES as FAILURE_ANALYSIS_CLASSES


EXPECTED_CLASSES = [
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motor",
    "bike",
    "traffic light",
    "traffic sign",
]


class BDD100KCategoryMappingTest(unittest.TestCase):
    def test_all_consumers_use_converter_category_order(self):
        self.assertEqual(DATASET_CLASSES, EXPECTED_CLASSES)
        self.assertEqual(VISUALIZATION_CLASSES, EXPECTED_CLASSES)
        self.assertEqual(ANALYSIS_CLASSES, EXPECTED_CLASSES)
        self.assertEqual(FAILURE_ANALYSIS_CLASSES, EXPECTED_CLASSES)
        self.assertEqual(
            WEATHER_CATEGORIES,
            [
                {"id": index, "name": name}
                for index, name in enumerate(EXPECTED_CLASSES, start=1)
            ],
        )


if __name__ == "__main__":
    unittest.main()
