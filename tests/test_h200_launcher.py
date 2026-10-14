"""CPU-only tests for the standalone H200 bootstrap (no torch import)."""

import ast
import contextlib
import copy
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("h200_launcher", ROOT / "scripts/run_h200.py")
h200 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h200)


def checkpoint():
    recipe = copy.deepcopy(h200.RECIPE_DEFAULTS)
    recipe.update(epochs=12, batch_size=2, accum_steps=4, lr=3e-5,
                  lr_steps=[7, 10], warmup_iters=750)
    return {
        "epoch": 5, "args": recipe,
        "state_keys": ["model_state_dict", "optimizer_state_dict",
                       "scheduler_state_dict", "scaler_state_dict"],
        "metadata": {
            "architecture": {
                "model_name": "fastvit_sa12", "architecture_version": "fixed_c5",
                "num_classes": 10, "fpn_channels": 256,
                "anchor_sizes": [11, 19, 28, 51, 153],
                "regression_mode": "box_delta", "reg_max": None,
            },
            "training": {
                "classification_loss": "qfl", "classification_beta": 2.0,
                "classification_alpha": None, "regression_loss": "ciou",
                "matcher": "fixed_iou_0.4_0.5_force_positive",
            },
        },
    }


class H200LauncherTest(unittest.TestCase):
    def resume_args(self, *extra):
        return h200.parse_args(["--mode", "resume", "--checkpoint", "last.pth", *extra])

    def assert_parse_error(self, argv):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            h200.parse_args(argv)

    def test_mode_defaults_and_single_gpu(self):
        a = self.resume_args()
        self.assertEqual((a.branch, a.mode), ("refactor-bdd", "resume"))
        self.assertEqual(a.work_dir, Path("/datastore/locth/ChienAnh").resolve())
        self.assertEqual(a.venv, a.work_dir / ".venv")
        self.assertEqual(a.data_dir, a.work_dir / "data/bdd100k")
        self.assertEqual(a.batch_size * a.accum_steps, 8)
        self.assertEqual((a.batch_size, a.accum_steps, a.eval_batch_size, a.workers),
                         (2, 4, 2, 2))
        self.assert_parse_error(["--checkpoint", "last.pth", "--gpu", "0,1"])
        self.assert_parse_error(["--mode", "resume"])
        self.assert_parse_error(["--mode", "train", "--checkpoint", "last.pth"])
        self.assert_parse_error(["--checkpoint", "last.pth", "--from-scratch"])
        self.assert_parse_error(["--checkpoint", "last.pth", "--epochs", "0"])

    def test_reject_secrets_and_ref_options(self):
        self.assert_parse_error(["--checkpoint", "last.pth", "--repo-url",
                                 "https://token@github.com/user/repo.git"])
        self.assert_parse_error(["--checkpoint", "last.pth", "--branch=--upload-pack=x"])
        self.assert_parse_error(["--checkpoint", "last.pth", "--commit", "shortsha"])

    def test_dry_run_has_no_writes_or_subprocesses(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "not-created"
            with patch.object(h200.subprocess, "run", side_effect=AssertionError("subprocess")), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                code = h200.main(["--mode", "resume", "--checkpoint", "missing.pth", "--work-dir", str(target),
                                  "--dry-run"])
            self.assertEqual(code, 0)
            self.assertFalse(target.exists())
            self.assertIn("refactor-bdd", output.getvalue())
            self.assertIn("--save-json", output.getvalue())

    def test_resume_inherits_recipe_not_launcher_defaults(self):
        a = self.resume_args()
        recipe = h200.checkpoint_recipe(a, checkpoint())
        self.assertEqual(recipe["lr"], 3e-5)
        self.assertEqual(recipe["lr_steps"], [7, 10])
        argv = h200.training_command(a, Path("/venv/bin/python"), Path("/repo"), recipe)
        self.assertEqual(argv[argv.index("--lr") + 1], "3e-05")
        self.assertEqual(argv[argv.index("--warmup-iters") + 1], "750")
        self.assertEqual(argv[argv.index("--epochs") + 1], "25")
        self.assertIn("--resume", argv)
        self.assertNotIn("--resume-weights-only", argv)
        self.assertIn("--amp", argv)

    def test_resume_checks_complete_states_and_target_epoch(self):
        c = checkpoint()
        c["state_keys"].remove("scheduler_state_dict")
        with self.assertRaisesRegex(ValueError, "state"):
            h200.checkpoint_recipe(self.resume_args(), c)
        with self.assertRaisesRegex(ValueError, "greater"):
            h200.checkpoint_recipe(self.resume_args("--epochs", "6"), checkpoint())

    def test_reject_training_metadata_mismatch_and_unsupported_architecture(self):
        c = checkpoint()
        c["metadata"]["training"]["classification_beta"] = 1.5
        with self.assertRaisesRegex(ValueError, "classification_beta"):
            h200.checkpoint_recipe(self.resume_args(), c)
        c = checkpoint()
        c["metadata"]["architecture"]["reg_max"] = 16
        with self.assertRaisesRegex(ValueError, "reg_max"):
            h200.checkpoint_recipe(self.resume_args(), c)

    def test_cosine_schedule_is_not_silently_extended(self):
        c = checkpoint()
        c["args"].update(scheduler="cosine", epochs=50)
        with self.assertRaisesRegex(ValueError, "cosine"):
            h200.checkpoint_recipe(self.resume_args("--epochs", "12"), c)
        recipe = h200.checkpoint_recipe(self.resume_args("--epochs", "50"), c)
        self.assertEqual(recipe["scheduler"], "cosine")

    def test_eval_does_not_require_optimizer_and_can_evaluate_completed_target(self):
        c = checkpoint()
        c["state_keys"] = ["model_state_dict"]
        a = self.resume_args("--mode", "eval", "--epochs", "1", "--score-thresh", "0.001")
        argv = h200.training_command(a, Path("/venv/python"), Path("/repo"),
                                     h200.checkpoint_recipe(a, c))
        self.assertIn("--eval-only", argv)
        self.assertEqual(argv[argv.index("--score-thresh") + 1], "0.001")

    def test_fresh_training_is_explicit(self):
        a = h200.parse_args(["--from-scratch", "--model", "fastvit_t8"])
        self.assertEqual((a.mode, a.epochs), ("train", 25))
        recipe = h200.checkpoint_recipe(a, None)
        self.assertEqual(recipe["model"], "fastvit_t8")
        argv = h200.training_command(a, Path("/venv/python"), Path("/repo"), recipe)
        self.assertNotIn("--resume", argv)
        self.assertNotIn("--pretrained-backbone", argv)

    def test_default_training_with_pretrained_backbone_starts_fresh(self):
        a = h200.parse_args(["--pretrained-backbone", "imagenet.pth"])
        self.assertEqual(a.mode, "train")
        self.assertIsNone(a.checkpoint)
        argv = h200.training_command(a, Path("/venv/python"), Path("/repo"),
                                     h200.checkpoint_recipe(a, None))
        self.assertIn("--pretrained-backbone", argv)
        self.assertNotIn("--resume", argv)
        self.assert_parse_error(["--checkpoint", "last.pth"])

    def test_default_training_selects_cached_imagenet_backbone(self):
        a = h200.parse_args([])
        self.assertEqual((a.mode, a.epochs, a.auto_pretrained), ("train", 25, True))
        self.assertIsNone(a.checkpoint)
        self.assertEqual(a.pretrained_backbone,
                         a.work_dir / "checkpoints/fastvit_sa12_imagenet_unfused.pth.tar")
        self.assertFalse(self.resume_args().auto_pretrained)
        self.assert_parse_error(["--model", "fastvit_t8"])

    def test_data_audit_checks_category_mapping_and_sample_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            a = self.resume_args("--mode", "eval", "--data-dir", temp)
            root = Path(temp) / "images/100k/val"
            root.mkdir(parents=True)
            (root / "sample.jpg").touch()
            ann = Path(temp) / "annotations/bdd100k_det_val_coco.json"
            ann.parent.mkdir()
            data = {"categories": [{"id": i, "name": n}
                                   for i, n in enumerate(h200.CLASSES, 1)],
                    "images": [{"id": 1, "file_name": "sample.jpg"}], "annotations": []}
            ann.write_text(json.dumps(data), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(h200.audit_data(a)["val"]["images"], 1)
            data["categories"][0]["name"] = "wrong-class"
            ann.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "canonical"):
                h200.audit_data(a)
            data["categories"][0]["name"] = "person"
            data["images"][0]["file_name"] = "../outside.jpg"
            ann.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsafe"):
                h200.audit_data(a)

    def test_existing_clone_is_not_modified_when_dirty(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp) / "source"
            repo.mkdir()
            a = SimpleNamespace(work_dir=Path(temp), repo_url=h200.REPO_URL)
            with patch.object(h200, "command",
                              side_effect=[str(repo), h200.REPO_URL, " M important.py"]) as run:
                with self.assertRaisesRegex(ValueError, "local changes"):
                    h200.clone_source(a, {})
            self.assertEqual(run.call_count, 3)
            self.assertNotIn("checkout", [x for call in run.call_args_list for x in call.args[0]])

    def test_generated_flags_exist_in_current_trainer(self):
        # No importing train_bdd100k.py: this check must run without torch.
        tree = ast.parse((ROOT / "train_bdd100k.py").read_text(encoding="utf-8"))
        supported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "add_argument":
                supported.update(arg.value for arg in node.args
                                 if isinstance(arg, ast.Constant) and isinstance(arg.value, str))
        supported.update({"--no-amp", "--no-wandb"})
        a = self.resume_args("--mode", "eval", "--max-eval-samples", "10")
        argv = h200.training_command(a, Path("/python"), Path("/repo"), h200.RECIPE_DEFAULTS)
        self.assertTrue({x for x in argv if x.startswith("--")}.issubset(supported))

    def test_stream_job_propagates_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            fake = SimpleNamespace(stdout=io.StringIO("out of memory\n"),
                                   wait=lambda: 17)
            with patch.object(h200.subprocess, "Popen", return_value=fake), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = h200.stream_job(["python", "train.py"], Path(temp), {},
                                       Path(temp) / "train.log")
            self.assertEqual(code, 17)
            self.assertIn("out of memory", (Path(temp) / "train.log").read_text())


if __name__ == "__main__":
    unittest.main()
