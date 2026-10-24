"""Resume submission checks with a fake launcher and a saved-recipe fixture."""

import contextlib
import io
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import test_h200_shell as shell_tests


ROOT = shell_tests.ROOT
BASH = shell_tests.BASH
SCRIPT = ROOT / "scripts/resume_h200.sh"


@unittest.skipUnless(BASH, "Bash not installed")
class H200ResumeTest(unittest.TestCase):
    def invoke(self, *args, env=None):
        return subprocess.run(
            [BASH, SCRIPT.as_posix(), *args], cwd=ROOT,
            capture_output=True, text=True, timeout=20,
            env=shell_tests.shell_environment(**(env or {})))

    def test_syntax_header_and_lf(self):
        result = subprocess.run([BASH, "-n", SCRIPT.as_posix()],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(b"\r", SCRIPT.read_bytes())
        header, body = SCRIPT.read_text().split("set -euo pipefail", 1)
        self.assertNotIn("#SBATCH", body)
        directives = dict(line.removeprefix("#SBATCH --").split("=", 1)
                          for line in header.splitlines() if line.startswith("#SBATCH --"))
        self.assertEqual(directives["output"], directives["error"])
        self.assertEqual(directives["output"],
                         "/datastore/locth/ChienAnh/logs/fastvit_resume_%j.log")
        self.assertEqual(directives["gres"], "gpu:2g.35gb:1")
        self.assertEqual(directives["time"], "72:00:00")

    def test_defaults_are_full_resume_to_total_50(self):
        result = self.invoke("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = shlex.split(next(line[2:] for line in result.stdout.splitlines()
                                if line.startswith("+ ")))
        self.assertEqual(argv[:2], ["bash", "/datastore/locth/ChienAnh/run_h200.sh"])
        self.assertEqual(argv[argv.index("--mode") + 1], "resume")
        self.assertEqual(argv[argv.index("--checkpoint") + 1],
                         "/datastore/locth/ChienAnh/runs/fastvit_sa12_20261017_165236/last.pth")
        for flag, value in (("--epochs", "50"), ("--batch-size", "4"),
                            ("--accum-steps", "4"), ("--eval-batch-size", "2"),
                            ("--workers", "2")):
            self.assertEqual(argv[argv.index(flag) + 1], value)
        self.assertNotIn("--from-scratch", argv)
        self.assertNotIn("--pretrained-backbone", argv)
        self.assertIn("--epoch-log-only", argv)
        self.assertIn("--post-train-eval", argv)

    def test_dry_run_is_write_free_and_uses_scheduler_log(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp) / "not created"
            result = self.invoke("--work-dir", work.as_posix(), "--dry-run",
                                 env={"SLURM_JOB_ID": "98765",
                                      "CUDA_VISIBLE_DEVICES": "MIG-assigned"})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(work.exists())
            self.assertIn("/logs/fastvit_resume_98765.log", result.stdout)

    def run_fixture(self, root, exit_code=0, launcher=True, checkpoint=True, extra=()):
        work = root / "persistent work"
        (work / "runs/fastvit_sa12_20261017_165236").mkdir(parents=True)
        if checkpoint:
            (work / "runs/fastvit_sa12_20261017_165236/last.pth").write_bytes(b"fixture")
        if launcher:
            (work / "run_h200.sh").write_text(
                '#!/usr/bin/env bash\n'
                'printf "%s\\0" "$@" > "$RESUME_TEST_CAPTURE/args"\n'
                'printf "%s" "$CUDA_VISIBLE_DEVICES" > "$RESUME_TEST_CAPTURE/gpu"\n'
                'printf "trainer stdout\\n"\nprintf "trainer stderr\\n" >&2\n'
                'exit "$RESUME_TEST_EXIT"\n', encoding="utf-8", newline="\n")
        # sbatch spools just this file; the launcher must be found under work-dir.
        spool = root / "slurm spool"
        spool.mkdir()
        copy = spool / "job_script"
        shutil.copyfile(SCRIPT, copy)
        command = (
            'resume_work="$(cd "$2" && pwd -P)"\n'
            'export RESUME_TEST_CAPTURE="$resume_work"\n'
            'exec bash "$1" --work-dir "$resume_work" '
            '--log-file "$resume_work/runtime log.log" "${@:3}"\n')
        result = subprocess.run(
            [BASH, "-c", command, "--", copy.as_posix(), work.as_posix(), *extra],
            capture_output=True, text=True, timeout=20,
            env=shell_tests.shell_environment(
                RESUME_TEST_EXIT=str(exit_code), CUDA_VISIBLE_DEVICES="MIG-assigned"))
        capture = work / "args"
        argv = capture.read_bytes().decode().split("\0")[:-1] if capture.exists() else None
        return result, argv, work

    def test_spooled_resume_preserves_gpu_arguments_logs_and_failure_status(self):
        with tempfile.TemporaryDirectory() as temp:
            result, argv, work = self.run_fixture(
                Path(temp), exit_code=17,
                extra=("--batch-size", "2", "--accum-steps", "8", "--prepare-only",
                       "--venv", "/venvs/custom env", "--data-dir", "/data/bdd 100k",
                       "--no-epoch-log-only"))
            self.assertEqual(result.returncode, 17)
            self.assertEqual((result.stdout, result.stderr), ("", ""))
            self.assertEqual(argv[argv.index("--mode") + 1], "resume")
            self.assertEqual(argv[argv.index("--epochs") + 1], "50")
            self.assertEqual(argv[argv.index("--venv") + 1], "/venvs/custom env")
            self.assertEqual(argv[argv.index("--data-dir") + 1], "/data/bdd 100k")
            # The main launcher's parser uses the last override.
            last_batch = len(argv) - 1 - argv[::-1].index("--batch-size")
            self.assertEqual(argv[last_batch + 1], "2")
            self.assertIn("--prepare-only", argv)
            self.assertGreater(argv.index("--no-epoch-log-only"), argv.index("--epoch-log-only"))
            self.assertEqual((work / "gpu").read_text(), "MIG-assigned")
            text = (work / "runtime log.log").read_text()
            self.assertIn("trainer stdout", text)
            self.assertIn("trainer stderr", text)
            self.assertTrue((work / "runs/fastvit_sa12_20261017_165236/last.pth").exists())

    def test_missing_inputs_fail_before_calling_launcher(self):
        for launcher, checkpoint, message in (
                (False, True, "Missing launcher"),
                (True, False, "Checkpoint missing or empty")):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temp:
                result, argv, work = self.run_fixture(
                    Path(temp), launcher=launcher, checkpoint=checkpoint)
                self.assertEqual(result.returncode, 2)
                self.assertEqual((result.stdout, result.stderr), ("", ""))
                self.assertIsNone(argv)
                self.assertIn(message, (work / "runtime log.log").read_text())

    def test_invalid_options_cannot_switch_to_new_training(self):
        for args in (("--mode", "train"), ("--from-scratch",),
                     ("--pretrained-backbone", "imagenet.pth"),
                     ("--epochs", "0"), ("--checkpoint",)):
            with self.subTest(args=args):
                result = self.invoke(*args, "--dry-run")
                self.assertEqual(result.returncode, 2)

    def test_checkpoint_after_32_epochs_keeps_recipe_and_has_18_remaining(self):
        case = shell_tests.H200ShellPythonTest()
        checkpoint = case.full_checkpoint()
        checkpoint["epoch"] = 31
        checkpoint["args"].update(epochs=50, batch_size=4, accum_steps=4,
                                  lr=1e-4, lr_steps=[8, 11])
        checkpoint["scheduler_state_dict"] = {"iteration": 60000, "epoch": 31}
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            report, argv = case.run_recipe(temp, checkpoint, epochs="50")
        self.assertEqual(50 - (report["checkpoint"]["epoch"] + 1), 18)
        self.assertEqual(report["recipe"]["lr"], 1e-4)
        self.assertEqual(report["recipe"]["lr_steps"], [8, 11])
        self.assertEqual(report["recipe"]["warmup_iters"], 750)
        self.assertIn("--amp", argv)


if __name__ == "__main__":
    unittest.main()
