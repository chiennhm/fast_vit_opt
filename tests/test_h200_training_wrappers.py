"""Exercise cloned training wrappers with a fake Python, never CUDA/training."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_h200_shell import BASH, ROOT, SCRIPT, shell_environment


@unittest.skipUnless(BASH, "Bash not installed")
class H200TrainingWrapperTest(unittest.TestCase):
    def invoke(self, entrypoint, args, exit_code=0):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "source clone"
            (repo / "scripts").mkdir(parents=True)
            (root / "outside").mkdir()
            for name in ("train.sh", "resume.sh"):
                shutil.copyfile(ROOT / "scripts" / name, repo / "scripts" / name)
            (repo / "train_bdd100k.py").touch()
            python = root / "venv with spaces/bin/python"
            python.parent.mkdir(parents=True)
            python.write_text(
                '#!/usr/bin/env bash\nset -euo pipefail\n'
                'printf "%s\\0" "$@" > "$FASTVIT_TEST_CAPTURE/args"\n'
                'pwd -P > "$FASTVIT_TEST_CAPTURE/cwd"\n'
                'printf "%s" "$CUDA_VISIBLE_DEVICES" > "$FASTVIT_TEST_CAPTURE/gpu"\n'
                'exit "$FASTVIT_TEST_EXIT"\n', encoding="utf-8", newline="\n")
            python.chmod(0o755)
            command = (
                'work="$(cd "$1" && pwd -P)"\n'
                'export FASTVIT_TEST_CAPTURE="$work"\n'
                'export FASTVIT_PYTHON="$work/venv with spaces/bin/python"\n'
                'cd "$work/outside"\n'
                'exec bash "$work/source clone/scripts/$2" "${@:3}"\n'
            )
            result = subprocess.run(
                [BASH, "-c", command, "--", root.as_posix(), entrypoint, *args],
                capture_output=True, text=True, timeout=20,
                env=shell_environment(FASTVIT_TEST_EXIT=str(exit_code),
                                      CUDA_VISIBLE_DEVICES="MIG-assigned-by-slurm"))
            captured = root / "args"
            argv = captured.read_bytes().decode("utf-8").split("\0")[:-1] if captured.exists() else None
            cwd = (root / "cwd").read_text().strip() if (root / "cwd").exists() else None
            gpu = (root / "gpu").read_text() if (root / "gpu").exists() else None
            return result, argv, cwd, gpu

    def test_syntax_version_marker_and_lf(self):
        for name in ("train.sh", "resume.sh"):
            script = ROOT / "scripts" / name
            self.assertNotIn(b"\r", script.read_bytes())
            self.assertIn("# FASTVIT_TRAINER_ARGS_V1", script.read_text())
            result = subprocess.run([BASH, "-n", script.as_posix()],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_train_passthrough_keeps_exact_recipe_and_venv(self):
        flags = ["--model", "fastvit_sa12", "--epochs", "50", "--batch-size", "4",
                 "--accum-steps", "4", "--workers", "2", "--lr", "0.0001",
                 "--lr-steps", "8", "11", "--no-amp",
                 "--pretrained-backbone", "/weights/ImageNet SA12.pth",
                 "--data-dir", "/data/bdd 100k", "--output", "/runs/custom dir",
                 "--save-json", "--no-wandb"]
        result, argv, cwd, gpu = self.invoke("train.sh", ["--", *flags])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((result.stdout, result.stderr), ("", ""))
        self.assertEqual(argv[:2], ["-u", cwd + "/train_bdd100k.py"])
        self.assertEqual(argv[2:], flags)
        self.assertTrue(cwd.endswith("/source clone"))
        self.assertEqual(gpu, "MIG-assigned-by-slurm")
        self.assertNotIn("--save-visualizations", argv)
        self.assertNotIn("--resume", argv)

    def test_resume_passthrough_keeps_checkpoint_and_preserves_failure(self):
        flags = ["--model", "fastvit_sa24", "--epochs", "50", "--batch-size", "4",
                 "--accum-steps", "4", "--scheduler", "step", "--lr-steps", "7", "10"]
        checkpoint = "/weights/last checkpoint.pth"
        result, argv, cwd, gpu = self.invoke("resume.sh", [checkpoint, "--", *flags], 17)
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertEqual((result.stdout, result.stderr), ("", ""))
        self.assertEqual(argv, ["-u", cwd + "/train_bdd100k.py",
                                *flags, "--resume", checkpoint])
        self.assertEqual(gpu, "MIG-assigned-by-slurm")
        self.assertNotIn("--save-visualizations", argv)

    def test_eval_uses_the_same_source_trainer_without_training_defaults(self):
        flags = ["--eval-only", "--eval-batch-size", "2", "--save-json"]
        result, argv, _, _ = self.invoke("resume.sh", ["/weights/best.pth", "--", *flags])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(argv[2:], [*flags, "--resume", "/weights/best.pth"])
        self.assertNotIn("--epochs", argv)
        self.assertNotIn("--pretrained-backbone", argv)

    def test_legacy_train_positionals_are_unchanged(self):
        result, argv, _, _ = self.invoke("train.sh", ["fastvit_t8", "6", "/data/local set"])
        self.assertEqual(result.returncode, 0, result.stderr)
        for key, value in (("--model", "fastvit_t8"), ("--batch-size", "6"),
                           ("--data-dir", "/data/local set"), ("--epochs", "50"),
                           ("--workers", "4"), ("--output", "./output/bdd100k")):
            self.assertEqual(argv[argv.index(key) + 1], value)
        self.assertIn("--save-visualizations", argv)

    def test_legacy_resume_positionals_are_unchanged(self):
        result, argv, _, _ = self.invoke(
            "resume.sh", ["/weights/last.pth", "3", "/data/local set", "fastvit_sa24"])
        self.assertEqual(result.returncode, 0, result.stderr)
        for key, value in (("--resume", "/weights/last.pth"), ("--model", "fastvit_sa24"),
                           ("--batch-size", "3"), ("--data-dir", "/data/local set"),
                           ("--output", "./output/bdd100k_resumed")):
            self.assertEqual(argv[argv.index(key) + 1], value)

    def test_help_and_missing_resume_never_invoke_python(self):
        for script in ("train.sh", "resume.sh"):
            result, argv, _, _ = self.invoke(script, ["--help"])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("FASTVIT_PYTHON", result.stdout)
            self.assertIsNone(argv)
        result, argv, _, _ = self.invoke("resume.sh", [])
        self.assertEqual(result.returncode, 2)
        self.assertIsNone(argv)

    def test_old_source_wrapper_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scripts = root / "source/scripts"
            scripts.mkdir(parents=True)
            (root / "source/train_bdd100k.py").touch()
            python = root / ".venv/bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("#!/usr/bin/env bash\nexit 99\n", newline="\n")
            python.chmod(0o755)
            old_wrapper = scripts / "train.sh"
            old_wrapper.write_text(
                '#!/usr/bin/env bash\nprintf "WRONG TRAINING STARTED\\n"\nexit 98\n',
                newline="\n")
            command = (
                'OSTYPE=linux-gnu\nwork="$(cd "$2" && pwd -P)"\n'
                'source "$1" --work-dir "$work" --from-scratch\n'
            )
            result = subprocess.run(
                [BASH, "-c", command, "--", SCRIPT.as_posix(), root.as_posix()],
                cwd=ROOT, capture_output=True, text=True, timeout=20, env=shell_environment())
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual((result.stdout, result.stderr), ("", ""))
            logs = list(root.glob("launches/*/run.log"))
            self.assertEqual(len(logs), 1)
            log = logs[0].read_text()
            self.assertIn("FASTVIT_TRAINER_ARGS_V1", log)
            self.assertIn("Push them to refactor-bdd", log)
            self.assertNotIn("WRONG TRAINING STARTED", log)


if __name__ == "__main__":
    unittest.main()
