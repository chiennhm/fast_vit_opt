"""CPU-only checks for the three-stage Bash H200 workflow."""

import ast
import contextlib
import copy
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test_h200_launcher import checkpoint, h200


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_h200.sh"
SOURCE = SCRIPT.read_text(encoding="utf-8")
SETUP_SCRIPT = ROOT / "scripts/setup_h200.sh"
DOWNLOAD_SCRIPT = ROOT / "scripts/download_h200.sh"
SETUP_SOURCE = SETUP_SCRIPT.read_text(encoding="utf-8")
DOWNLOAD_SOURCE = DOWNLOAD_SCRIPT.read_text(encoding="utf-8")
BASH = shutil.which("bash")
if BASH is None and os.name == "nt":
    for candidate in (Path("C:/Program Files/Git/bin/bash.exe"),
                      Path("C:/Program Files/Git/usr/bin/bash.exe")):
        if candidate.is_file():
            BASH = str(candidate)
            break


def embedded_python(marker):
    source = DOWNLOAD_SOURCE if marker == "PY_PRETRAINED" else SOURCE
    match = re.search(r"<<'" + marker + r"'\n(.*?)\n" + marker + r"\n", source, re.S)
    if match is None:
        raise AssertionError(f"Missing heredoc {marker}")
    return match.group(1)


def shell_environment(**overrides):
    # Tests must not accidentally inherit an enclosing cluster allocation.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("SLURM_") and k != "CUDA_VISIBLE_DEVICES"}
    env.update(overrides)
    return env


class H200ShellPythonTest(unittest.TestCase):
    def test_inline_python_syntax_and_lf(self):
        for script in (SCRIPT, SETUP_SCRIPT, DOWNLOAD_SCRIPT):
            self.assertNotIn(b"\r", script.read_bytes())
        for marker in ("PY_GPU", "PY_RECIPE", "PY_DATA", "PY_PRETRAINED", "PY_BACKBONE"):
            ast.parse(embedded_python(marker), filename=f"{SCRIPT}:{marker}")
        download_audit = re.search(r"<<'PY_DATA'\n(.*?)\nPY_DATA\n", DOWNLOAD_SOURCE, re.S)
        self.assertEqual(embedded_python("PY_DATA"), download_audit.group(1))

    def test_slurm_header_precedes_commands_and_combines_streams(self):
        header, body = SOURCE.split("set -euo pipefail", 1)
        self.assertNotIn("#SBATCH", body)
        directives = dict(line.removeprefix("#SBATCH --").split("=", 1)
                          for line in header.splitlines() if line.startswith("#SBATCH --"))
        self.assertEqual(directives["output"], directives["error"])
        self.assertEqual(directives["output"],
                         "/datastore/locth/ChienAnh/logs/fastvit_%j.log")
        self.assertEqual(directives["chdir"], "/datastore/locth/ChienAnh")
        self.assertEqual(directives["open-mode"], "append")
        self.assertEqual(directives["gres"], "gpu:2g.35gb:1")
        self.assertEqual(directives["ntasks"], "1")
        self.assertLessEqual(int(directives["cpus-per-task"]), 8)
        self.assertLessEqual(int(directives["mem"].removesuffix("G")), 64)
        self.assertLessEqual(int(directives["time"].split(":")[0]), 72)

    def run_recipe(self, temp, c=None, mode="resume", epochs="12"):
        args = ["-", mode, "trusted.pth" if c else "", epochs, "8", "1",
                "fastvit_sa12", temp]
        fake_torch = SimpleNamespace(load=lambda *a, **k: copy.deepcopy(c))
        with patch.object(sys, "argv", args), patch.dict(sys.modules, {"torch": fake_torch}), \
                contextlib.redirect_stdout(io.StringIO()):
            exec(compile(embedded_python("PY_RECIPE"), str(SCRIPT), "exec"), {})
        report = json.loads((Path(temp) / "recipe.json").read_text(encoding="utf-8"))
        argv = (Path(temp) / "recipe.args").read_bytes().decode("utf-8").split("\0")[:-1]
        return report, argv

    def test_new_recipe_matches_python_launcher(self):
        with tempfile.TemporaryDirectory() as temp:
            report, argv = self.run_recipe(temp, mode="train")
            self.assertEqual(report["recipe"], h200.RECIPE_DEFAULTS)
            self.assertIn("--amp", argv)

    def test_resume_preserves_lr_scheduler_and_loss(self):
        with tempfile.TemporaryDirectory() as temp:
            report, argv = self.run_recipe(temp, self.full_checkpoint())
            self.assertEqual(report["recipe"]["lr"], 3e-5)
            self.assertEqual(report["recipe"]["lr_steps"], [7, 10])
            self.assertEqual(argv[argv.index("--lr") + 1], "3e-05")
            self.assertIn("--classification-loss", argv)
            self.assertEqual(report["checkpoint"]["epoch"], 5)

    def test_resume_rejects_incomplete_state_and_completed_target(self):
        with tempfile.TemporaryDirectory() as temp:
            c = checkpoint()
            c["scheduler_state_dict"] = None
            with self.assertRaisesRegex(ValueError, "state"):
                self.run_recipe(temp, c)
            with self.assertRaisesRegex(ValueError, "exceed"):
                self.run_recipe(temp, self.full_checkpoint(), epochs="6")

    def test_long_resume_keeps_checkpoint_schedule_and_amp(self):
        c = self.full_checkpoint()
        c["args"].update(lr=1e-4, lr_steps=[8, 11], batch_size=2, accum_steps=4)
        with tempfile.TemporaryDirectory() as temp:
            report, argv = self.run_recipe(temp, c, epochs="25")
        self.assertEqual(report["recipe"]["lr"], 1e-4)
        self.assertEqual(report["recipe"]["lr_steps"], [8, 11])
        self.assertEqual(argv[argv.index("--lr-steps") + 1:argv.index("--lr-steps") + 3],
                         ["8", "11"])
        self.assertIn("--amp", argv)

    def full_checkpoint(self):
        c = checkpoint()
        for key in c["state_keys"]:
            c[key] = {}
        return c

    def test_rejects_metadata_mismatch(self):
        c = self.full_checkpoint()
        c["metadata"]["training"]["classification_beta"] = 1.5
        with tempfile.TemporaryDirectory() as temp, \
                self.assertRaisesRegex(ValueError, "classification_beta"):
            self.run_recipe(temp, c)

    def test_cosine_target_is_preserved(self):
        c = self.full_checkpoint()
        c["args"].update(scheduler="cosine", epochs=50)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "cosine"):
                self.run_recipe(temp, c)

    def test_eval_allows_weights_without_optimizer(self):
        c = checkpoint()
        with tempfile.TemporaryDirectory() as temp:
            report, _ = self.run_recipe(temp, c, mode="eval", epochs="1")
            self.assertEqual(report["checkpoint"]["epoch"], 5)

    def test_data_audit_detects_missing_image(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = root / "val"
            images.mkdir()
            (images / "sample.jpg").touch()
            ann = root / "val.json"
            data = {"categories": [{"id": i, "name": name}
                                   for i, name in enumerate(h200.CLASSES, 1)],
                    "images": [{"id": 1, "file_name": "sample.jpg"}], "annotations": []}
            ann.write_text(json.dumps(data), encoding="utf-8")
            args = ["-", "eval", "", "", str(images), str(ann), "8",
                    str(root / "dataset.json")]
            with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
                exec(compile(embedded_python("PY_DATA"), str(SCRIPT), "exec"), {})
            self.assertEqual(json.loads((root / "dataset.json").read_text())["val"]["images"], 1)
            data["images"][0]["file_name"] = "../missing.jpg"
            ann.write_text(json.dumps(data), encoding="utf-8")
            with patch.object(sys, "argv", args), self.assertRaisesRegex(ValueError, "unsafe"):
                exec(compile(embedded_python("PY_DATA"), str(SCRIPT), "exec"), {})


@unittest.skipUnless(BASH, "Bash not installed")
class H200ShellCliTest(unittest.TestCase):
    def run_shell(self, *args, env=None, mode="resume"):
        # Most existing tests exercise resume; fresh-run tests use mode=None
        # to test the real CLI default without injecting --mode.
        mode_args = ["--mode", mode] if mode else []
        return subprocess.run([BASH, SCRIPT.as_posix(), *mode_args, *args], cwd=ROOT,
                              capture_output=True, text=True, timeout=20,
                              env=shell_environment(**(env or {})))

    def slurm_environment(self, **overrides):
        env = {"SLURM_JOB_ID": "12345", "SLURM_CPUS_PER_TASK": "8",
               "CUDA_VISIBLE_DEVICES": "MIG-test-assigned-uuid"}
        env.update(overrides)
        return env

    def dry_run_args(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        return shlex.split(next(line[2:] for line in result.stdout.splitlines()
                                if line.startswith("+ ")))

    def test_bash_syntax_and_help(self):
        result = subprocess.run([BASH, "-n", SCRIPT.as_posix()], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.run_shell("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("refactor-bdd", result.stdout)

    def test_default_datastore_root_and_derived_paths(self):
        root = "/datastore/locth/ChienAnh"
        result = self.run_shell("--checkpoint", root + "/checkpoints/last.pth", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = shlex.split(next(line[2:] for line in result.stdout.splitlines()
                                if line.startswith("+ ")))
        self.assertEqual(argv[:4], ["env", "FASTVIT_PYTHON=" + root + "/.venv/bin/python",
                                  "bash", root + "/source/scripts/resume.sh"])
        self.assertEqual(argv[4:6], [root + "/checkpoints/last.pth", "--"])
        self.assertEqual(argv[argv.index("--data-dir") + 1], root + "/data/bdd100k")
        self.assertEqual(argv[argv.index("--epochs") + 1], "50")
        self.assertEqual(argv[argv.index("--train-img") + 1], root + "/data/bdd100k/images/100k/train")
        self.assertEqual(argv[argv.index("--output") + 1], root + "/runs")
        self.assertIn(root + "/launches/TIMESTAMP/run.log", result.stdout)

    def test_slurm_uses_mig_recipe_and_the_scheduler_log(self):
        result = self.run_shell("--checkpoint", "/checkpoints/last.pth", "--dry-run",
                                env=self.slurm_environment())
        argv = self.dry_run_args(result)
        for flag, value in (("--batch-size", "4"), ("--accum-steps", "4"),
                            ("--eval-batch-size", "2"), ("--workers", "2"),
                            ("--log-interval", "100")):
            self.assertEqual(argv[argv.index(flag) + 1], value)
        self.assertIn("SLURM mode: 1; effective batch: 16", result.stdout)
        self.assertIn("GPU: MIG-test-assigned-uuid", result.stdout)
        self.assertIn("/datastore/locth/ChienAnh/logs/fastvit_12345.log", result.stdout)
        self.assertNotIn("/launches/TIMESTAMP/run.log", result.stdout)

    def test_slurm_respects_recipe_and_log_overrides(self):
        result = self.run_shell(
            "--checkpoint", "last.pth", "--batch-size", "4", "--accum-steps", "2",
            "--eval-batch-size", "1", "--workers", "2", "--log-file", "/logs/custom.log",
            "--dry-run", env=self.slurm_environment())
        argv = self.dry_run_args(result)
        for flag, value in (("--batch-size", "4"), ("--accum-steps", "2"),
                            ("--eval-batch-size", "1"), ("--workers", "2")):
            self.assertEqual(argv[argv.index(flag) + 1], value)
        self.assertIn("effective batch: 8", result.stdout)
        self.assertIn("Runtime log (file only): /logs/custom.log", result.stdout)

    def test_slurm_worker_defaults_follow_cpu_allocation(self):
        for cpus, workers in (("8", "2"), ("4", "1"), ("2", "0"), ("1", "0")):
            with self.subTest(cpus=cpus):
                result = self.run_shell("--checkpoint", "last.pth", "--dry-run",
                                        env=self.slurm_environment(SLURM_CPUS_PER_TASK=cpus))
                argv = self.dry_run_args(result)
                self.assertEqual(argv[argv.index("--workers") + 1], workers)

    def test_slurm_rejects_workers_beyond_allocation(self):
        result = self.run_shell("--checkpoint", "last.pth", "--workers", "8", "--dry-run",
                                env=self.slurm_environment())
        self.assertEqual(result.returncode, 2)
        self.assertIn("CPU budget", result.stderr)
        for cpus in ("0", "-1", "bad"):
            with self.subTest(cpus=cpus):
                result = self.run_shell("--checkpoint", "last.pth", "--dry-run",
                                        env=self.slurm_environment(SLURM_CPUS_PER_TASK=cpus))
                self.assertEqual(result.returncode, 2)
                self.assertIn("SLURM_CPUS_PER_TASK", result.stderr)

    def test_slurm_rejects_gpu_override_and_missing_or_multiple_devices(self):
        result = self.run_shell("--checkpoint", "last.pth", "--gpu", "0", "--dry-run",
                                env=self.slurm_environment())
        self.assertEqual(result.returncode, 2)
        self.assertIn("Do not pass --gpu", result.stderr)
        for visible in ("", "-1", "NoDevFiles", "none", "0,1", "MIG-one,MIG-two"):
            with self.subTest(visible=visible):
                result = self.run_shell(
                    "--checkpoint", "last.pth", "--dry-run",
                    env=self.slurm_environment(CUDA_VISIBLE_DEVICES=visible))
                self.assertEqual(result.returncode, 2)
                self.assertIn("exactly one assigned", result.stderr)
        result = self.run_shell("--checkpoint", "last.pth", "--dry-run",
                                env={"SLURM_JOB_ID": "12345", "SLURM_CPUS_PER_TASK": "8"})
        self.assertEqual(result.returncode, 2)
        self.assertIn("exactly one assigned", result.stderr)

    def test_gpu_binding_preserves_scheduler_uuid_and_index(self):
        body = SOURCE.split("# BEGIN_GPU_ENVIRONMENT\n", 1)[1].split(
            "# END_GPU_ENVIRONMENT", 1)[0]
        setup = (
            'set -euo pipefail\nSLURM_MODE="$1"\nGPU=3\nGPU_EXPLICIT=0\n'
            'die() { printf "ERROR: %s\\n" "$*" >&2; exit 2; }\n'
        )
        for slurm, assigned, expected in (("1", "MIG-uuid-123", "MIG-uuid-123"),
                                          ("1", "2", "2"), ("1", "0", "0"),
                                          ("0", "old-device", "3")):
            with self.subTest(slurm=slurm, assigned=assigned):
                result = subprocess.run(
                    [BASH, "-c", setup + body + '\nprintf "%s" "$CUDA_VISIBLE_DEVICES"\n',
                     "--", slurm], capture_output=True, text=True, timeout=20,
                    env=shell_environment(CUDA_VISIBLE_DEVICES=assigned))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, expected)

    def test_dry_run_has_no_writes_and_quotes_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp) / "not-created"
            result = self.run_shell("--checkpoint", "/workspace/last checkpoint.pth",
                                    "--data-dir", "/workspace/data set",
                                    "--work-dir", work.as_posix(), "--dry-run")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(work.exists())
            self.assertIn("Offline run: prepared source", result.stdout)
            argv = shlex.split(next(line[2:] for line in result.stdout.splitlines()
                                    if line.startswith("+ ")))
            self.assertTrue(argv[3].endswith("/source/scripts/resume.sh"))
            self.assertEqual(argv[4:6], ["/workspace/last checkpoint.pth", "--"])
            self.assertEqual(argv[argv.index("--data-dir") + 1], "/workspace/data set")
            self.assertEqual(argv[argv.index("--batch-size") + 1], "4")
            self.assertEqual(argv[argv.index("--accum-steps") + 1], "4")

    def test_validation_errors_before_network_or_writes(self):
        bad = [
            [], ["--checkpoint"], ["--mode", "train", "--model", "fastvit_t8"],
            ["--checkpoint", "last.pth", "--gpu", "0,1"],
            ["--checkpoint", "last.pth", "--epochs", "0"],
            ["--checkpoint", "last.pth", "--batch-size", "-1"],
            ["--checkpoint", "last.pth", "--branch", "--bad"],
            ["--checkpoint", "last.pth", "--commit", "not-a-full-sha"],
            ["--checkpoint", "last.pth", "--repo-url", "https://token@github.com/repo.git"],
            ["--checkpoint", "last.pth", "--archive", "data.zip", "--download-data"],
            ["--checkpoint", "last.pth", "--download-data", "--no-download-data"],
            ["--checkpoint", "last.pth", "--unknown"],
        ]
        for args in bad:
            with self.subTest(args=args):
                result = self.run_shell(*args, "--dry-run")
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("ERROR:", result.stderr)

    def test_eval_and_new_training_modes(self):
        result = self.run_shell("--mode", "eval", "--checkpoint", "best.pth", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--eval-only", result.stdout)
        self.assertTrue(self.dry_run_args(result)[3].endswith("/source/scripts/resume.sh"))
        result = self.run_shell("--mode", "train", "--from-scratch", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--resume ", result.stdout)
        self.assertTrue(self.dry_run_args(result)[3].endswith("/source/scripts/train.sh"))
        self.assertEqual(self.dry_run_args(result)[4], "--")

    def test_default_new_training_has_no_checkpoint(self):
        result = self.run_shell("--from-scratch", "--dry-run", mode=None)
        argv = self.dry_run_args(result)
        self.assertIn("Mode: train; initialization: random weights", result.stdout)
        self.assertEqual(argv[argv.index("--epochs") + 1], "50")
        self.assertNotIn("--resume", argv)
        self.assertNotIn("--pretrained-backbone", argv)
        self.assertNotIn("--eval-only", argv)
        for extra in (["--checkpoint", "last.pth"],
                      ["--from-scratch", "--pretrained-backbone", "imagenet.pth"]):
            with self.subTest(extra=extra):
                result = self.run_shell(*extra, "--dry-run", mode=None)
                self.assertEqual(result.returncode, 2)

    def test_default_new_training_selects_imagenet_without_dry_run_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp) / "new-run"
            result = self.run_shell("--work-dir", work.as_posix(), "--dry-run", mode=None)
            argv = self.dry_run_args(result)
            self.assertIn("Mode: train", result.stdout)
            self.assertIn("--pretrained-backbone", argv)
            self.assertTrue(argv[argv.index("--pretrained-backbone") + 1].endswith(
                "/checkpoints/fastvit_sa12_backbone.pth"))
            self.assertNotIn("--resume", argv)
            self.assertFalse(work.exists())

    def test_default_new_training_can_initialize_only_backbone(self):
        result = self.run_shell("--pretrained-backbone", "/weights/ImageNet SA12.pth",
                                "--dry-run", mode=None, env=self.slurm_environment())
        argv = self.dry_run_args(result)
        self.assertIn("Mode: train", result.stdout)
        self.assertEqual(argv[argv.index("--pretrained-backbone") + 1],
                         "/weights/ImageNet SA12.pth")
        self.assertNotIn("--resume", argv)

    def test_custom_log_dry_run_does_not_create_log(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "logs" / "custom log.log"
            result = self.run_shell("--checkpoint", "last.pth", "--log-file",
                                    log.as_posix(), "--dry-run")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(log.exists())
            self.assertIn("Runtime log (file only)", result.stdout)

    def run_job_tail(self, temp, job_code):
        # Run the actual exit-code section, with a tiny fake training job.
        # No clone, pip, data download, CUDA or real trainer is invoked.
        tail = SOURCE[SOURCE.rindex("\nset +e\n") + 1:]
        setup = (
            'set -euo pipefail\nSOURCE_DIR="$1"\nLAUNCH_DIR="$1"\n'
            'exec >> "$LAUNCH_DIR/run.log" 2>&1\n'
            f'JOB=("$BASH" -c \'printf "test training log\\n"; '
            f'printf "test stderr log\\n" >&2; exit {job_code}\')\n'
        )
        return subprocess.run([BASH, "-c", setup + tail, "--", Path(temp).as_posix()],
                              capture_output=True, text=True, timeout=20,
                              env=shell_environment())

    def test_training_failure_is_logged_silently_with_correct_exit_code(self):
        with tempfile.TemporaryDirectory() as temp:
            result = self.run_job_tail(temp, job_code=17)
            self.assertEqual(result.returncode, 17, result.stderr)
            self.assertEqual((result.stdout, result.stderr), ("", ""))
            self.assertEqual((Path(temp) / "exit_code.txt").read_text().strip(), "17")
            log = (Path(temp) / "run.log").read_text()
            self.assertIn("test training log", log)
            self.assertIn("test stderr log", log)
            self.assertIn("finished with exit code: 17", log)
            self.assertFalse((Path(temp) / "train.log").exists())

    def test_successful_training_is_logged_silently(self):
        with tempfile.TemporaryDirectory() as temp:
            result = self.run_job_tail(temp, job_code=0)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((result.stdout, result.stderr), ("", ""))
            self.assertEqual((Path(temp) / "exit_code.txt").read_text().strip(), "0")
            self.assertIn("finished with exit code: 0", (Path(temp) / "run.log").read_text())


@unittest.skipUnless(BASH, "Bash not installed")
class H200ShellLoggingTest(unittest.TestCase):
    def run_logging(self, temp, log="", exit_code=0):
        body = SOURCE.split("# BEGIN_FILE_LOGGING\n", 1)[1].split(
            "# END_FILE_LOGGING", 1)[0]
        setup = r"""
set -euo pipefail
WORK_DIR="$(cd "$1" && pwd -P)"
LOG_FILE="$2"
if [[ -n "$LOG_FILE" ]]; then
  LOG_FILE="$(cd "$(dirname -- "$LOG_FILE")" && pwd -P)/$(basename -- "$LOG_FILE")"
fi
MODE=resume
BRANCH=refactor-bdd
GPU=0
"""
        output = (
            '\nprintf "setup stdout\\n"\nprintf "setup stderr\\n" >&2\n'
            '\nprintf "training stdout\\n"\nprintf "training stderr\\n" >&2\n'
            f'exit {exit_code}\n'
        )
        return subprocess.run([BASH, "-c", setup + body + output, "--",
                               Path(temp).as_posix(), log],
                              capture_output=True, text=True, timeout=20,
                              env=shell_environment())

    def test_all_runtime_streams_use_one_default_log(self):
        with tempfile.TemporaryDirectory() as temp:
            result = self.run_logging(temp, exit_code=23)
            self.assertEqual(result.returncode, 23, result.stderr)
            self.assertEqual((result.stdout, result.stderr), ("", ""))
            logs = list(Path(temp).glob("launches/*/run.log"))
            self.assertEqual(len(logs), 1)
            text = logs[0].read_text()
            for line in ("setup stdout", "setup stderr", "training stdout", "training stderr"):
                self.assertIn(line, text)
            self.assertLess(SOURCE.index("# BEGIN_FILE_LOGGING"),
                            SOURCE.index('[[ "$OSTYPE" == linux* ]]'))

    def test_custom_log_is_append_only_and_handles_spaces(self):
        with tempfile.TemporaryDirectory() as temp:
            logfile = Path(temp) / "existing custom.log"
            logfile.write_text("previous run\n")
            result = self.run_logging(temp, logfile.as_posix())
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((result.stdout, result.stderr), ("", ""))
            text = logfile.read_text()
            self.assertTrue(text.startswith("previous run\n"))
            self.assertIn("training stderr", text)
            self.assertEqual(list(Path(temp).glob("launches/*/run.log")), [])


class H200PretrainedPreparationTest(unittest.TestCase):
    class Tensor:
        def __init__(self, shape=(2, 2)):
            self.shape = shape

    def run_prepare(self, root, state=None, auto=True, redownload=False, offline=False, save_failure=False):
        expected = {"patch_embed.weight": self.Tensor(), "network.0.weight": self.Tensor(),
                    "norm0.weight": self.Tensor()}
        if state is None:
            state = {"state_dict": {"module.patch_embed.weight": self.Tensor(),
                                    "module.network.0.weight": self.Tensor(),
                                    "module.head.weight": self.Tensor()}}
        recorded = {}
        backbone = SimpleNamespace(
            state_dict=lambda: expected, named_parameters=lambda: expected.items(),
            load_state_dict=lambda values, strict: recorded.update(loaded=values))
        def save(values, path):
            recorded["saved"] = values
            Path(path).write_bytes(b"normalized fixture")
            if save_failure:
                raise OSError("failed to save normalized weights")
        fake_torch = SimpleNamespace(load=lambda *a, **k: state, save=save,
                                     is_tensor=lambda value: isinstance(value, self.Tensor))
        raw = root / "cache/sa12.pth.tar"
        output = root / "launch/pretrained_backbone.pth"
        report = root / "launch/pretrained.json"
        args = ["-", str(raw), str(output), str(report), "1" if auto else "0", "fastvit_sa12",
                "1" if redownload else "0"]
        if offline:
            args = ["-", str(raw), str(output), str(report), "fastvit_sa12"]
        fake_modules = {"torch": fake_torch, "timm": SimpleNamespace(),
                        "timm.models": SimpleNamespace(create_model=lambda *a, **k: backbone),
                        "models": SimpleNamespace()}
        with patch.dict(sys.modules, fake_modules), patch.object(sys, "argv", args), \
                patch.object(urllib.request, "urlopen", return_value=io.BytesIO(b"official fixture")) as fetch, \
                contextlib.redirect_stdout(io.StringIO()):
            marker = "PY_BACKBONE" if offline else "PY_PRETRAINED"
            exec(compile(embedded_python(marker), str(SCRIPT), "exec"), {})
        return json.loads(report.read_text()), recorded, fetch

    def test_download_then_reuse_and_keep_only_backbone_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report, recorded, fetch = self.run_prepare(root)
            self.assertEqual(fetch.call_count, 1)
            self.assertIn("fastvit_sa12.pth.tar", fetch.call_args.args[0])
            self.assertEqual(set(recorded["saved"]["state_dict"]),
                             {"patch_embed.weight", "network.0.weight"})
            self.assertEqual(report["matched_core_parameters"], 2)
            self.assertEqual(report["fresh_backbone_keys"], ["norm0.weight"])
            self.assertEqual(report["start_epoch"], 0)
            self.assertTrue(report["fresh_optimizer_scheduler_scaler"])
            self.assertEqual(list((root / "cache").glob("*.part")), [])
            _, _, fetch = self.run_prepare(root)
            self.assertEqual(fetch.call_count, 0)

    def test_detector_checkpoint_and_incompatible_backbone_are_rejected(self):
        cases = [
            ({"model_state_dict": {}}, "detector checkpoint"),
            ({"state_dict": {"patch_embed.weight": self.Tensor()}}, "core parameters missing"),
            ({"state_dict": {"patch_embed.weight": self.Tensor((1,)),
                             "network.0.weight": self.Tensor()}}, "core parameters missing"),
        ]
        for state, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                with self.assertRaisesRegex(ValueError, message):
                    self.run_prepare(root, state)
                self.assertFalse((root / "launch/pretrained_backbone.pth").exists())

    def test_missing_user_weights_are_not_automatically_downloaded(self):
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(FileNotFoundError):
            self.run_prepare(Path(temp), auto=False)

    def test_forced_refresh_replaces_cache_only_after_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "cache/sa12.pth.tar"
            raw.parent.mkdir()
            raw.write_bytes(b"old valid checkpoint")
            with self.assertRaisesRegex(ValueError, "core parameters missing"):
                self.run_prepare(root, {"state_dict": {}}, redownload=True)
            self.assertEqual(raw.read_bytes(), b"old valid checkpoint")
            self.assertEqual(list(raw.parent.glob("*.part")), [])
            report, _, fetch = self.run_prepare(root, redownload=True)
            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(raw.read_bytes(), b"official fixture")
            self.assertTrue(report["forced_redownload"])

    def test_failed_download_removes_partial_file_and_never_starts_pretraining(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(shutil, "copyfileobj", side_effect=OSError("network failure")), \
                    self.assertRaisesRegex(OSError, "network failure"):
                self.run_prepare(root)
            self.assertFalse((root / "cache/sa12.pth.tar").exists())
            self.assertEqual(list((root / "cache").glob("*.part")), [])


    def test_run_backbone_validation_is_offline_and_filters_classifier(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "cache/sa12.pth.tar"
            raw.parent.mkdir()
            raw.write_bytes(b"cached fixture")
            report, recorded, fetch = self.run_prepare(root, offline=True)
            fetch.assert_not_called()
            self.assertEqual(set(recorded["saved"]["state_dict"]),
                             {"patch_embed.weight", "network.0.weight"})
            self.assertTrue(report["fresh_detection_head"])
            self.assertTrue(report["fresh_optimizer_scheduler_scaler"])
            with self.assertRaisesRegex(ValueError, "detector checkpoint"):
                self.run_prepare(root, {"model_state_dict": {}}, offline=True)

    def test_failed_prepared_save_preserves_both_existing_weight_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "cache/sa12.pth.tar"
            prepared = root / "launch/pretrained_backbone.pth"
            raw.parent.mkdir()
            prepared.parent.mkdir()
            raw.write_bytes(b"old raw")
            prepared.write_bytes(b"old prepared")
            with self.assertRaisesRegex(OSError, "failed to save"):
                self.run_prepare(root, redownload=True, save_failure=True)
            self.assertEqual(raw.read_bytes(), b"old raw")
            self.assertEqual(prepared.read_bytes(), b"old prepared")
            self.assertEqual(list(root.rglob("*.part")), [])


@unittest.skipUnless(BASH, "Bash not installed")
class H200ShellDataPreparationTest(unittest.TestCase):
    def run_prepare(self, temp, mode="resume", action="auto", custom=False):
        body = DOWNLOAD_SOURCE.split("# BEGIN_DATA_PREPARATION\n", 1)[1].split(
            "# END_DATA_PREPARATION", 1)[0]
        setup = r"""
set -euo pipefail
SOURCE_DIR="$1"
DATA_DIR="$1/data"
MODE="$2"
DATA_ACTION="$3"
CUSTOM_DATA_PATHS="$4"
ARCHIVE="$1/input.zip"
TRAIN_IMG="$DATA_DIR/images/100k/train"
VAL_IMG="$DATA_DIR/images/100k/val"
TRAIN_ANN="$DATA_DIR/annotations/bdd100k_det_train_coco.json"
VAL_ANN="$DATA_DIR/annotations/bdd100k_det_val_coco.json"
ENV_PYTHON=mock_converter
die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
mock_converter() { printf '%s\0' "$@" > "$SOURCE_DIR/converter.args"; }
"""
        result = subprocess.run(
            [BASH, "-c", setup + body, "--", Path(temp).as_posix(), mode, action,
             "1" if custom else "0"],
            capture_output=True, text=True, timeout=20, env=shell_environment())
        recorded = Path(temp) / "converter.args"
        args = recorded.read_bytes().decode("utf-8").split("\0")[:-1] if recorded.exists() else None
        return result, args

    def make_images(self, root, splits=("train", "val")):
        for split in splits:
            shard = root / "data/images/100k" / split / "nested shard"
            shard.mkdir(parents=True)
            (shard / "sample.JPG").touch()

    def make_annotations(self, root, splits=("train", "val")):
        ann = root / "data/annotations"
        ann.mkdir(parents=True, exist_ok=True)
        for split in splits:
            (ann / f"bdd100k_det_{split}_coco.json").write_text('{"images": []}', encoding="utf-8")

    def test_prepared_data_skips_converter_without_completion_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_images(root)
            self.make_annotations(root)
            result, args = self.run_prepare(temp)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIsNone(args)
            self.assertIn("Dataset already present", result.stdout)
            self.assertFalse((root / "data/.kaggle_bdd100k_complete").exists())
            self.assertEqual((root / "data/annotations/bdd100k_det_train_coco.json").read_text(),
                             '{"images": []}')

    def test_missing_data_selects_download_and_keeps_zip(self):
        with tempfile.TemporaryDirectory() as temp:
            result, args = self.run_prepare(temp)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(args[:2], ["-m", "tools.download_bdd100k"])
            self.assertIn("--keep-zip", args)
            self.assertNotIn("--archive", args)
            self.assertNotIn("--convert-only", args)
            self.assertIn("downloading BDD100K", result.stdout)

    def test_raw_data_converts_without_download(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_images(root)
            labels = root / "data/bdd100k_labels_release/bdd100k/labels"
            labels.mkdir(parents=True)
            for split in ("train", "val"):
                (labels / f"bdd100k_labels_images_{split}.json").write_text("[]")
            result, args = self.run_prepare(temp)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--convert-only", args)
            self.assertIn("no download", result.stdout)

    def test_local_zip_is_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data").mkdir()
            (root / "data/bdd100k.zip").touch()
            result, args = self.run_prepare(temp)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--archive", args)
            self.assertTrue(args[args.index("--archive") + 1].endswith("/data/bdd100k.zip"))
            self.assertIn("no download", result.stdout)

    def test_no_download_option_does_not_call_converter(self):
        with tempfile.TemporaryDirectory() as temp:
            result, args = self.run_prepare(temp, action="--no-download-data")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIsNone(args)
            self.assertIn("disabled", result.stdout)

    def test_eval_reuses_only_validation_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_images(root, ("val",))
            self.make_annotations(root, ("val",))
            result, args = self.run_prepare(temp, mode="eval")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIsNone(args)

    def test_incomplete_custom_paths_do_not_trigger_download(self):
        with tempfile.TemporaryDirectory() as temp:
            result, args = self.run_prepare(temp, custom=True)
            self.assertEqual(result.returncode, 2)
            self.assertIsNone(args)
            self.assertIn("Custom image/annotation paths", result.stderr)

    def test_explicit_download_still_skips_existing_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_images(root)
            self.make_annotations(root)
            result, args = self.run_prepare(temp, action="--download-data")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIsNone(args)

    def test_explicit_conversion_is_honored(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_images(root)
            self.make_annotations(root)
            result, args = self.run_prepare(temp, action="--convert-only")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--convert-only", args)



@unittest.skipUnless(BASH, "Bash not installed")
class H200SplitStagesTest(unittest.TestCase):
    def invoke(self, script, *args, env=None):
        return subprocess.run([BASH, script.as_posix(), *args], cwd=ROOT,
                              capture_output=True, text=True, timeout=20,
                              env=shell_environment(**(env or {})))

    def test_all_scripts_have_valid_bash_syntax_and_help(self):
        for script in (SETUP_SCRIPT, DOWNLOAD_SCRIPT, SCRIPT):
            with self.subTest(script=script.name):
                result = subprocess.run([BASH, "-n", script.as_posix()],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.invoke(script, "--help").returncode, 0)

    def test_stages_have_distinct_side_effect_boundaries(self):
        # The GPU job consumes prepared local artifacts; no fetch/install/download.
        for forbidden in ("git clone", 'fetch origin', 'checkout --detach',
                          "-m pip install", "urllib", "tools.download_bdd100k"):
            self.assertNotIn(forbidden, SOURCE)
        for forbidden in ("torch.cuda", "CUDA_VISIBLE_DEVICES", "PY_GPU", "#SBATCH"):
            self.assertNotIn(forbidden, SETUP_SOURCE)
            self.assertNotIn(forbidden, DOWNLOAD_SOURCE)
        self.assertNotIn("tools.download_bdd100k", SETUP_SOURCE)
        self.assertNotIn("urllib", SETUP_SOURCE)
        self.assertNotIn("-m pip install", DOWNLOAD_SOURCE)
        self.assertNotIn("fetch origin", DOWNLOAD_SOURCE)
        # No sourcing relative to script location: sbatch may spool just run_h200.sh.
        self.assertNotIn("BASH_SOURCE", SOURCE)
        self.assertNotRegex(SOURCE, r"(?m)^\s*(?:source|\.) ")
        self.assertIn("Next: bash download_h200.sh", SETUP_SOURCE)
        self.assertIn("Next: sbatch run_h200.sh", DOWNLOAD_SOURCE)

    def test_setup_and_download_dry_run_without_gpu_or_writes(self):
        for script in (SETUP_SCRIPT, DOWNLOAD_SCRIPT):
            with self.subTest(script=script.name), tempfile.TemporaryDirectory() as temp:
                work = Path(temp) / "new work"
                log = work / "log dir/custom.log"
                result = self.invoke(script, "--work-dir", work.as_posix(), "--log-file",
                                     log.as_posix(), "--dry-run",
                                     env={"SLURM_JOB_ID": "999", "CUDA_VISIBLE_DEVICES": ""})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(work.exists())
                self.assertIn("DRY RUN", result.stdout)
                self.assertIn("custom.log", result.stdout)

    def test_setup_defaults_and_validation(self):
        result = self.invoke(SETUP_SCRIPT, "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("refactor-bdd", result.stdout)
        self.assertIn("https://github.com/chiennhm/fast_vit_opt.git", result.stdout)
        self.assertIn("/datastore/locth/ChienAnh/.venv", result.stdout)
        for args in (["--branch", "../bad"], ["--commit", "bad"],
                     ["--repo-url", "https://secret@github.com/x/y.git"], ["--venv"]):
            with self.subTest(args=args):
                result = self.invoke(SETUP_SCRIPT, *args, "--dry-run")
                self.assertEqual(result.returncode, 2)
        result = self.invoke(SETUP_SCRIPT, "--commit", "a" * 40, "--skip-install", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("skip install: 1", result.stdout)

    def test_download_modes_and_argument_validation(self):
        for args in ([], ["--pretrained-only", "--redownload-pretrained"],
                     ["--data-only"], ["--mode", "eval", "--data-only"],
                     ["--archive", "/data/local.zip"], ["--convert-only"],
                     ["--no-download-data"], ["--pretrained-backbone", "/weights/sa12.pth"]):
            with self.subTest(args=args):
                result = self.invoke(DOWNLOAD_SCRIPT, *args, "--dry-run")
                self.assertEqual(result.returncode, 0, result.stderr)
        for args in (["--data-only", "--pretrained-only"],
                     ["--data-only", "--redownload-pretrained"],
                     ["--pretrained-backbone", "/weights/sa12.pth", "--redownload-pretrained"],
                     ["--model", "../unsafe"], ["--model", "fastvit_t8"],
                     ["--mode", "resume"], ["--archive", "data.zip", "--download-data"],
                     ["--convert-only", "--no-download-data"]):
            with self.subTest(args=args):
                self.assertEqual(self.invoke(DOWNLOAD_SCRIPT, *args, "--dry-run").returncode, 2)

    def test_run_rejects_moved_flags_with_actionable_stage_names(self):
        for flag, owner in (("--branch", "setup_h200.sh"), ("--skip-install", "setup_h200.sh"),
                            ("--redownload-pretrained", "download_h200.sh"),
                            ("--download-data", "download_h200.sh"),
                            ("--archive", "download_h200.sh")):
            with self.subTest(flag=flag):
                result = self.invoke(SCRIPT, flag, "--dry-run")
                self.assertEqual(result.returncode, 2)
                self.assertIn(owner, result.stderr)

    def test_missing_setup_fails_silently_to_file_before_gpu_or_network(self):
        # Override OSTYPE solely for the test so Git Bash can exercise Linux preflight.
        for script in (DOWNLOAD_SCRIPT, SCRIPT):
            with self.subTest(script=script.name), tempfile.TemporaryDirectory() as temp:
                command = 'OSTYPE=linux-gnu; work="$(cd "$2" && pwd -P)"; source "$1" --work-dir "$work"'
                result = subprocess.run(
                    [BASH, "-c", command, "--", script.as_posix(), Path(temp).as_posix()],
                    cwd=ROOT, capture_output=True, text=True, timeout=20, env=shell_environment())
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertEqual((result.stdout, result.stderr), ("", ""))
                logs = list(Path(temp).glob("**/*.log"))
                self.assertEqual(len(logs), 1)
                self.assertIn("Run setup_h200.sh first", logs[0].read_text())


if __name__ == "__main__":
    unittest.main()
