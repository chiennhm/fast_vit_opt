"""Check that epoch-only output leaves training/evaluation behavior intact."""

import argparse
import ast
import contextlib
import importlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from detection.losses import DetectionLoss
from engine.train import train_one_epoch
from test_train import TinyDetector


class EpochLoggingTest(unittest.TestCase):
    def test_quiet_training_has_identical_losses_and_updates(self):
        batch = (torch.zeros(1, 3, 16, 16), [{
            "boxes": torch.tensor([[0., 0., 10., 10.]]),
            "labels": torch.tensor([1]),
        }])
        quiet_model, verbose_model = TinyDetector(), TinyDetector()
        quiet_optimizer = torch.optim.SGD(quiet_model.parameters(), lr=0.01)
        verbose_optimizer = torch.optim.SGD(verbose_model.parameters(), lr=0.01)
        stderr = io.StringIO()
        with self.assertNoLogs("engine.train", level="INFO"), contextlib.redirect_stderr(stderr):
            quiet_metrics = train_one_epoch(
                quiet_model, DetectionLoss(num_classes=1), [batch, batch, batch],
                quiet_optimizer, None, torch.device("cpu"), 32,
                amp=False, log_interval=0, accum_steps=2)
        self.assertEqual(stderr.getvalue(), "")
        with self.assertLogs("engine.train", level="INFO"), \
                contextlib.redirect_stderr(io.StringIO()):
            verbose_metrics = train_one_epoch(
                verbose_model, DetectionLoss(num_classes=1), [batch, batch, batch],
                verbose_optimizer, None, torch.device("cpu"), 32,
                amp=False, log_interval=1, accum_steps=2)
        self.assertEqual(quiet_metrics, verbose_metrics)
        for quiet, verbose in zip(quiet_model.parameters(), verbose_model.parameters()):
            self.assertTrue(torch.equal(quiet, verbose))
        self.assertFalse(torch.equal(quiet_model.logits, torch.zeros_like(quiet_model.logits)))

    def test_quiet_training_keeps_exceptions_visible_to_caller(self):
        model = TinyDetector()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = Mock(side_effect=RuntimeError("fixture CUDA out of memory"))
        with self.assertRaisesRegex(RuntimeError, "fixture CUDA out of memory"):
            train_one_epoch(
                model, criterion, [(torch.zeros(1, 3, 16, 16), [])],
                optimizer, None, torch.device("cpu"), 32,
                amp=False, log_interval=0)

    def test_evaluation_without_progress_still_returns_metrics_and_predictions(self):
        evaluation = importlib.import_module("engine.evaluate")

        class Predictor(torch.nn.Module):
            def predict(self, images, **kwargs):
                return [{"boxes": torch.tensor([[0., 0., 10., 10.]]),
                         "scores": torch.tensor([0.9]),
                         "labels": torch.tensor([1])} for _ in images]

        target = {"boxes": torch.tensor([[0., 0., 10., 10.]]),
                  "labels": torch.tensor([1]), "size": torch.tensor([16, 16]),
                  "orig_size": torch.tensor([32, 32]), "image_id": torch.tensor(1)}
        batch = (torch.zeros(1, 3, 16, 16), [target])
        stderr = io.StringIO()
        with patch.object(evaluation, "evaluate_coco",
                          return_value={"mAP": 0.3, "coco_protocol": {}}) as metric, \
                contextlib.redirect_stderr(stderr):
            result = evaluation.evaluate(
                Predictor(), [batch], torch.device("cpu"),
                class_names=["person"], amp=False, show_progress=False)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(result["mAP"], 0.3)
        self.assertEqual(result["num_images"], 1)
        predictions, targets = metric.call_args.args
        self.assertEqual(predictions[0]["boxes"].tolist(), [[0., 0., 20., 20.]])
        self.assertEqual(targets[0]["boxes"].tolist(), [[0., 0., 20., 20.]])

    def trainer_functions(self):
        # Isolate CLI/wiring from the ImageNet backbone dependency and CUDA.
        source = Path(__file__).resolve().parents[1] / "train_bdd100k.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("parse_args", "_run_evaluation")]
        scope = {"argparse": argparse, "evaluate": Mock(return_value={"mAP": 0.3}),
                 "BDD100K_CLASSES": ["person"]}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), scope)
        return scope

    def test_flag_disables_both_train_and_eval_progress_and_allows_opt_out(self):
        scope = self.trainer_functions()
        for flags, interval in ((["--epoch-log-only"], 0),
                                (["--epoch-log-only", "--no-epoch-log-only"], 100)):
            with self.subTest(flags=flags), patch.object(sys, "argv", ["train", *flags]):
                args = scope["parse_args"]()
                self.assertEqual(args.log_interval, interval)
                scope["_run_evaluation"](None, [], torch.device("cpu"), args, None)
                self.assertEqual(scope["evaluate"].call_args.kwargs["show_progress"],
                                 interval > 0)

    def test_invalid_interval_or_conflicting_debug_is_rejected(self):
        scope = self.trainer_functions()
        for flags in (["--log-interval", "-1"],
                      ["--epoch-log-only", "--debug-first-batch"]):
            with self.subTest(flags=flags), patch.object(sys, "argv", ["train", *flags]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                scope["parse_args"]()


if __name__ == "__main__":
    unittest.main()
