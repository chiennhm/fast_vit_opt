"""Final evaluation uses canonical COCO matching and one inference cache."""

import ast
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from detection import metrics
from engine.evaluation_suite import (
    filter_predictions, prepare_weather_groups, run_evaluation_suite,
    select_final_checkpoint,
)


def fixture_records():
    predictions = [
        {"boxes": np.array([[0, 0, 10, 10], [0, 0, 10, 10],
                            [22, 22, 30, 30], [0, 0, 10, 10]], dtype=np.float32),
         "scores": np.array([.9, .8, .7, .6], dtype=np.float32),
         "labels": np.array([1, 1, 1, 2]), "image_id": np.array(101),
         "orig_size": np.array([64, 64])},
        {"boxes": np.array([[40, 40, 50, 50]], dtype=np.float32),
         "scores": np.array([.4], dtype=np.float32), "labels": np.array([1]),
         "image_id": np.array(909), "orig_size": np.array([64, 64])},
    ]
    targets = [
        {"boxes": np.array([[0, 0, 10, 10], [20, 20, 40, 40]], dtype=np.float32),
         "labels": np.array([1, 1]), "iscrowd": np.array([0, 1]),
         "image_id": np.array(101), "orig_size": np.array([64, 64])},
        {"boxes": np.zeros((0, 4), dtype=np.float32),
         "labels": np.zeros(0, dtype=np.int64), "iscrowd": np.zeros(0, dtype=np.int64),
         "image_id": np.array(909), "orig_size": np.array([64, 64])},
    ]
    return predictions, targets


class WeatherAndCheckpointTest(unittest.TestCase):
    def write_raw(self, root, rows):
        raw = root / "labels/det_20/det_val.json"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text(json.dumps(rows), encoding="utf-8")
        return raw

    def test_weather_join_uses_filename_with_shards_and_arbitrary_image_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_raw(root, [
                {"name": "b.jpg", "attributes": {"weather": "unknown"}},
                {"name": "a.jpg", "attributes": {"weather": "partly cloudy"}},
            ])
            groups, source = prepare_weather_groups([
                {"id": 900, "file_name": "shard/a.jpg"},
                {"id": 7, "file_name": "b.jpg"},
            ], root)
            self.assertEqual(groups, {"partly_cloudy": [900], "undefined": [7]})
            self.assertTrue(source.endswith("det_val.json"))

    def test_missing_weather_and_ambiguous_names_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = [{"id": 1, "file_name": "a.jpg"}]
            self.write_raw(root, [{"name": "a.jpg", "attributes": {}}])
            with self.assertRaisesRegex(ValueError, "attributes.weather"):
                prepare_weather_groups(image, root)
            self.write_raw(root, [{"name": "b.jpg", "attributes": {"weather": "rainy"}}])
            with self.assertRaisesRegex(ValueError, "no weather metadata"):
                prepare_weather_groups(image, root)
            self.write_raw(root, [
                {"name": "one/a.jpg", "attributes": {"weather": "clear"}},
                {"name": "two/a.jpg", "attributes": {"weather": "clear"}},
            ])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                prepare_weather_groups(image, root)

    def test_resume_without_new_best_finds_original_best(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original, resumed, current = (root / name for name in ("original", "resumed", "current"))
            for folder in (original, resumed, current):
                folder.mkdir()
            torch.save({"args": {}}, original / "last.pth")
            torch.save({"args": {}}, original / "best.pth")
            torch.save({"args": {"resume": str(original / "last.pth")}}, resumed / "last.pth")
            best, reason = select_final_checkpoint(current, resumed / "last.pth")
            self.assertEqual(best, original / "best.pth")
            self.assertEqual(reason, "best_in_resume_ancestry")
            torch.save({"args": {}}, current / "best.pth")
            self.assertEqual(select_final_checkpoint(current, resumed / "last.pth"),
                             (current / "best.pth", "best_in_current_run"))

    def test_last_fallback_and_resume_cycle(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            torch.save({"args": {}}, root / "last.pth")
            with self.assertLogs("engine.evaluation_suite", level="WARNING"):
                selected, reason = select_final_checkpoint(root)
            self.assertEqual(selected, root / "last.pth")
            self.assertIn("fallback", reason)
            torch.save({"args": {"resume": str(root / "last.pth")}}, root / "last.pth")
            with self.assertRaisesRegex(ValueError, "Cycle"):
                select_final_checkpoint(root / "new", root / "last.pth")

    def test_filter_is_strict_and_keeps_empty_images(self):
        predictions, _ = fixture_records()
        predictions[0]["scores"] = np.array([.9, .5, .5, .5], dtype=np.float64)
        result = filter_predictions(predictions, .5)
        self.assertEqual(result[0]["scores"].tolist(), [.9])
        self.assertEqual(result[1]["boxes"].shape, (0, 4))
        self.assertEqual(len(predictions[0]["scores"]), 4)

    def test_trainer_loads_selected_best_weights_and_records_actual_epoch(self):
        source = Path(__file__).resolve().parents[1] / "train_bdd100k.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("_run_final_evaluation", "_dataset_paths")]
        load = Mock()
        scope = {"Path": Path, "torch": torch, "load_checkpoint": load,
                 "_checkpoint_training_contract": lambda args: {},
                 "_sha256": lambda path: "checkpoint-sha", "_git_output": lambda *args: "eval-sha",
                 "BDD100K_CLASSES": ["person"]}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), scope)
        args = SimpleNamespace(
            resume="original/last.pth", data_dir="/data/bdd100k", val_ann=None, val_img=None,
            train_ann=None, train_img=None, img_size=800, score_thresh=.05,
            confidence_thresholds=[.01, .05, .5], amp=True, nms_thresh=.5,
            max_detections=100, pre_nms_topk=2000, max_eval_samples=None,
            model="fastvit_sa12", allow_missing_checkpoint_metadata=False)
        model = SimpleNamespace(architecture_metadata=lambda: {"model_name": "fastvit_sa12"})
        loader = SimpleNamespace(dataset=SimpleNamespace(images={1: {"id": 1, "file_name": "a.jpg"}}))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            selected = root / "best.pth"
            torch.save({"epoch": 42, "best_map": .4,
                        "metadata": {"source": {"commit": "training-sha"}}}, selected)
            for explicit in (False, True):
                with self.subTest(explicit=explicit), \
                        patch("engine.evaluation_suite.select_final_checkpoint",
                              return_value=(selected, "best_in_current_run")) as choose, \
                        patch("engine.evaluation_suite.run_evaluation_suite",
                              return_value={"complete": True}) as suite:
                    scope["_run_final_evaluation"](
                        model, loader, torch.device("cpu"), args, root,
                        ({"clear": [1]}, "raw_val.json"),
                        checkpoint_path=selected if explicit else None)
                    self.assertTrue(load.call_args.kwargs["weights_only"])
                    self.assertEqual(load.call_args.args[0], selected)
                    options = suite.call_args.kwargs
                    self.assertEqual(options["checkpoint_info"]["completed_epochs"], 43)
                    self.assertEqual(options["checkpoint_info"]["sha256"], "checkpoint-sha")
                    self.assertEqual(options["checkpoint_info"]["preprocessing"]["short_side"], 800)
                    self.assertEqual(options["reference_confidence"], .05)
                    self.assertEqual(options["confidence_thresholds"], [.01, .05, .5])
                    self.assertEqual(options["output_dir"], root / "post_training_evaluation")
                    if explicit:
                        choose.assert_not_called()
                        self.assertEqual(options["checkpoint_info"]["selection"], "explicit_eval_checkpoint")


@unittest.skipUnless(metrics.HAS_PYCOCOTOOLS, "pycocotools is required for COCO integration")
class FinalCocoSuiteTest(unittest.TestCase):
    def test_counts_include_negative_images_duplicate_fp_and_ignore_crowd(self):
        predictions, targets = fixture_records()
        result = metrics.evaluate_coco(
            predictions, targets, num_classes=2, class_names=["person", "rider"],
            include_operating_points=True)
        point = result["operating_point"]
        self.assertEqual((point["TP"], point["FP"], point["FN"]), (1, 3, 0))
        self.assertAlmostEqual(point["precision"], .25)
        self.assertAlmostEqual(point["recall"], 1.)
        self.assertAlmostEqual(point["f1"], .4)
        self.assertEqual(result["operating_points_by_image"]["909"]["FP"], 1)
        self.assertIsNone(result["per_class"]["rider"]["AP"])
        self.assertIsNone(result["per_class"]["rider"]["operating_point"]["recall"])

    def test_zero_predictions_and_no_gt_cases(self):
        predictions, targets = fixture_records()
        empty = filter_predictions(predictions, 1.0)
        result = metrics.evaluate_coco(
            empty, targets, num_classes=2, class_names=["person", "rider"],
            include_operating_points=True)
        self.assertEqual(result["operating_point"]["FN"], 1)
        self.assertEqual(result["operating_point"]["FP"], 0)
        self.assertEqual(result["operating_point"]["f1"], 0.)
        result = metrics.evaluate_coco(
            predictions[1:], targets[1:], num_classes=2, class_names=["person", "rider"],
            include_operating_points=True)
        self.assertEqual(result["operating_point"]["FP"], 1)
        self.assertIsNone(result["operating_point"]["recall"])
        self.assertIsNone(result["operating_point"]["f1"])

    def test_real_cpu_inference_and_all_reports_use_one_pass(self):
        predictions, targets = fixture_records()

        class Predictor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
                self.cutoffs = []

            def predict(self, images, **kwargs):
                self.calls += 1
                self.cutoffs.append(kwargs["score_thresh"])
                return [{key: torch.as_tensor(row[key])
                         for key in ("boxes", "scores", "labels")} for row in predictions]

        model = Predictor()
        torch_targets = []
        for target in targets:
            target = {key: torch.as_tensor(value) for key, value in target.items()}
            target["size"] = torch.tensor([64, 64])
            torch_targets.append(target)
        loader = [(torch.zeros(2, 3, 64, 64), torch_targets)]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = run_evaluation_suite(
                model, loader, torch.device("cpu"), class_names=["person", "rider"],
                output_dir=root, weather_groups={"rainy": [101], "clear": [909]},
                source_images=[{"id": 101, "file_name": "rain.jpg"},
                               {"id": 909, "file_name": "clear.jpg"}],
                checkpoint_info={"path": "/trusted/best.pth", "selection": "fixture",
                                 "completed_epochs": 50},
                reference_confidence=.05, confidence_thresholds=[.001, .5, .85, .95],
                amp=False, max_samples=None,
            )
            self.assertEqual(model.calls, 1)
            self.assertEqual(model.cutoffs, [.001])
            self.assertEqual(manifest["evaluated_images"], 2)
            self.assertFalse(manifest["subset_evaluation"])
            self.assertEqual(manifest["weather_image_counts"], {"clear": 1, "rainy": 1})
            with (root / "metrics_by_confidence.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            low = next(row for row in rows if float(row["Confidence"]) == .05)
            high = next(row for row in rows if float(row["Confidence"]) == .85)
            self.assertEqual(float(low["Precision50"]), 25.)
            self.assertEqual(float(low["Recall50"]), 100.)
            self.assertEqual(float(low["F1_50"]), 40.)
            self.assertEqual(float(low["FP_per_image"]), 1.5)
            self.assertEqual(float(high["Precision50"]), 100.)
            self.assertEqual(float(high["FP_per_image"]), 0.)
            with (root / "metrics_by_weather.csv").open(newline="") as handle:
                weather = {row["Weather"]: row for row in csv.DictReader(handle)}
            self.assertEqual(weather["clear"]["AP"], "")
            self.assertEqual(weather["clear"]["Recall50"], "")
            gt = json.loads((root / "inference_cache/ground_truths_BDD100K.json").read_text())
            self.assertEqual(len(gt["images"]), 2)
            self.assertEqual(gt["images"][1]["file_name"], "clear.jpg")
            self.assertTrue((root / "confidence_analysis.png").stat().st_size > 1000)
            self.assertTrue((root / "REPORT.md").exists())
            with (root / "metrics_by_weather_confidence.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2 * len(rows))


if __name__ == "__main__":
    unittest.main()
