#!/usr/bin/env python3
"""Standalone Linux bootstrap for one-GPU FastViT/BDD100K jobs.

Copy this file to the server first; it does not require an existing clone.
Only the Python standard library is imported by the launcher.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


REPO_URL = "https://github.com/chiennhm/fast_vit_opt.git"
BRANCH = "refactor-bdd"
CLASSES = [
    "person", "rider", "car", "truck", "bus", "train", "motor", "bike",
    "traffic light", "traffic sign",
]
ALIASES = {"pedestrian": "person", "motorcycle": "motor", "bicycle": "bike"}
RECIPE_DEFAULTS = {
    "model": "fastvit_sa12", "architecture_version": "fixed_c5",
    "fpn_channels": 256, "anchor_sizes": [11, 19, 28, 51, 153],
    "img_size": 800, "lr": 1e-4, "weight_decay": 0.05,
    "clip_grad": 5.0, "classification_loss": "qfl",
    "focal_alpha": 0.25, "focal_gamma": 2.0, "scheduler": "step",
    "warmup_iters": 500, "warmup_epochs": 5, "lr_steps": [8, 11],
    "lr_gamma": 0.1, "seed": 42, "amp": True,
}

GPU_PROBE = """
import json, torch, torchvision
from torchvision.ops import nms
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable; refusing CPU fallback")
p = torch.cuda.get_device_properties(0)
x = torch.randn(32, 32, device="cuda")
assert torch.isfinite(x @ x).all().item()
boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device="cuda")
assert nms(boxes, torch.tensor([.9, .8], device="cuda"), .5).numel() == 1
torch.cuda.synchronize()
print(json.dumps({"name": p.name, "memory_gib": p.total_memory / 2**30,
    "capability": [p.major, p.minor], "torch": torch.__version__,
    "torchvision": torchvision.__version__, "cuda": torch.version.cuda}))
"""

CHECKPOINT_PROBE = """
import json, sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if not isinstance(c, dict):
    raise ValueError("Expected a training checkpoint dictionary")
print(json.dumps({"epoch": c.get("epoch"), "args": c.get("args"),
    "metadata": c.get("metadata"), "best_map": c.get("best_map"),
    "state_keys": [k for k in (
        "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
        "scaler_state_dict") if c.get(k) is not None]}))
"""

PRETRAINED_PREPARE = r"""
import json, os, shutil, sys, tempfile, urllib.request
from pathlib import Path
import torch
from timm.models import create_model
import models  # noqa: F401, registers the cloned FastViT variants

raw_path, output_path, report_path, auto_download, model_name, redownload = sys.argv[1:]
raw_path, output_path, report_path = map(Path, (raw_path, output_path, report_path))
url = "https://docs-assets.developer.apple.com/ml-research/models/fastvit/image_classification_models/fastvit_sa12.pth.tar"
if auto_download == "1" and model_name != "fastvit_sa12":
    raise ValueError("Automatic ImageNet weights are configured for fastvit_sa12 only")
if redownload == "1" and auto_download != "1":
    raise ValueError("Redownload applies only to the automatic ImageNet cache")
temp_path = None
try:
    candidate = raw_path
    if not raw_path.is_file() or redownload == "1":
        if auto_download != "1":
            raise FileNotFoundError(f"Missing trusted ImageNet backbone: {raw_path}")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=raw_path.parent, suffix=".part", delete=False) as handle:
            temp_path = Path(handle.name)
            print(f"Downloading official unfused ImageNet weights: {url}", flush=True)
            with urllib.request.urlopen(url, timeout=60) as response:
                shutil.copyfileobj(response, handle)
        if temp_path.stat().st_size == 0:
            raise ValueError("Downloaded ImageNet checkpoint is empty")
        candidate = temp_path
    else:
        print(f"Reusing local ImageNet backbone: {raw_path}", flush=True)

    # Official Apple weights or a trusted user-provided ImageNet checkpoint.
    checkpoint = torch.load(candidate, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" in checkpoint:
        raise ValueError("Expected ImageNet backbone weights, not a trained detector checkpoint")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError("ImageNet checkpoint state_dict must be a mapping")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    backbone = create_model(model_name, fork_feat=True, feature_version="fixed_c5")
    expected = backbone.state_dict()
    matched = {key: value for key, value in state.items()
               if key in expected and torch.is_tensor(value) and value.shape == expected[key].shape}
    # ImageNet has no detector output norms; those may initialize fresh.
    # All patch embedding/stage parameters must load, preventing silent partial/zero loads.
    required = {key for key, _ in backbone.named_parameters()
                if key.startswith(("patch_embed.", "network."))}
    missing = sorted(required - matched.keys())
    if not required or missing:
        raise ValueError(f"Incompatible/unfused backbone required: {len(missing)} core parameters missing; "
                         f"examples: {missing[:8]}")
    backbone.load_state_dict(matched, strict=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Separate tensor-only artifact for the trainer; never carry detection-head/optimizer state.
    torch.save({"state_dict": matched}, output_path)
    # A forced refresh replaces the cache only after download and compatibility checks pass.
    if temp_path is not None and (redownload == "1" or not raw_path.exists()):
        os.replace(temp_path, raw_path)
    report = {"source_path": str(raw_path), "source_url": url if auto_download == "1" else None,
              "downloaded": temp_path is not None, "forced_redownload": redownload == "1",
              "trainer_path": str(output_path), "model": model_name,
              "initialization": "imagenet_backbone_only", "start_epoch": 0,
              "matched_state_keys": len(matched), "total_backbone_state_keys": len(expected),
              "matched_core_parameters": len(required),
              "fresh_backbone_keys": sorted(expected.keys() - matched.keys()),
              "fresh_detection_head": True, "fresh_optimizer_scheduler_scaler": True}
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
finally:
    if temp_path is not None:
        temp_path.unlink(missing_ok=True)
"""


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["train", "resume", "eval"], default="train")
    p.add_argument("--repo-url", default=REPO_URL)
    p.add_argument("--branch", default=BRANCH)
    p.add_argument("--commit", help="Optional full 40-character Git commit to pin")
    p.add_argument("--work-dir", type=Path, default=Path("/datastore/locth/ChienAnh"))
    p.add_argument("--venv", type=Path, help="Default: WORK_DIR/.venv")
    p.add_argument("--skip-install", action="store_true", help="Reuse a prepared venv")
    p.add_argument("--checkpoint", type=Path, help="Trusted local checkpoint on server")
    init = p.add_mutually_exclusive_group()
    init.add_argument("--pretrained-backbone", type=Path)
    init.add_argument("--from-scratch", action="store_true")
    p.add_argument("--redownload-pretrained", action="store_true",
                   help="Refresh the official ImageNet cache from Apple's URL")
    p.add_argument("--model", choices=[
        "fastvit_t8", "fastvit_t12", "fastvit_s12", "fastvit_sa12",
        "fastvit_sa24", "fastvit_sa36", "fastvit_ma36",
    ], default="fastvit_sa12", help="Used only for a new training run")
    p.add_argument("--data-dir", type=Path, help="Default: WORK_DIR/data/bdd100k")
    for name in ("train-img", "train-ann", "val-img", "val-ann"):
        p.add_argument("--" + name, type=Path)
    data = p.add_mutually_exclusive_group()
    data.add_argument("--download-data", action="store_true",
                      help="Explicitly allow the large Kaggle download")
    data.add_argument("--archive", type=Path, help="Prepare an existing BDD100K ZIP")
    data.add_argument("--convert-only", action="store_true")
    p.add_argument("--epochs", type=int, default=25, help="Total target epochs, NOT extra")
    # Match the previous Kaggle run, including micro-batch size.
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--gpu", default="0", help="One CUDA device index or GPU UUID")
    p.add_argument("--allow-other-gpu", action="store_true")
    p.add_argument("--score-thresh", type=float, default=0.05)
    p.add_argument("--nms-thresh", type=float, default=0.5)
    p.add_argument("--max-detections", type=int, default=100)
    p.add_argument("--pre-nms-topk", type=int, default=2000)
    p.add_argument("--max-eval-samples", type=int, help="Smoke test only; not a benchmark")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print plan; no writes/network/GPU")
    a = p.parse_args(argv)
    if a.mode != "train" and not a.checkpoint:
        p.error("--mode resume/eval requires --checkpoint")
    if a.mode == "train" and a.checkpoint:
        p.error("Use --mode resume for --checkpoint; train starts a new run")
    if a.mode != "train" and (a.pretrained_backbone or a.from_scratch):
        p.error("backbone initialization options apply only to --mode train")
    a.auto_pretrained = a.mode == "train" and not a.pretrained_backbone and not a.from_scratch
    if a.auto_pretrained and a.model != "fastvit_sa12":
        p.error("Automatic ImageNet weights support SA12; pass --pretrained-backbone or --from-scratch")
    if a.redownload_pretrained and not a.auto_pretrained:
        p.error("--redownload-pretrained requires new training with the automatic ImageNet backbone")
    if a.workers < 0 or any(getattr(a, k) < 1 for k in (
            "epochs", "batch_size", "accum_steps", "eval_batch_size",
            "max_detections", "pre_nms_topk")):
        p.error("sizes/epochs must be positive and workers must be nonnegative")
    if a.max_eval_samples is not None and a.max_eval_samples < 1:
        p.error("--max-eval-samples must be positive")
    if not 0 <= a.score_thresh <= 1 or not 0 < a.nms_thresh <= 1:
        p.error("invalid score/NMS threshold")
    if not a.gpu or "," in a.gpu:
        p.error("this trainer supports one GPU per job, not DDP")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", a.branch) or ".." in a.branch:
        p.error("invalid branch")
    if a.commit and not re.fullmatch(r"[0-9a-fA-F]{40}", a.commit):
        p.error("--commit must be a full 40-character SHA")
    url = urlsplit(a.repo_url)
    if not a.repo_url.startswith("git@") and (
            url.scheme != "https" or not url.hostname or url.username or
            url.password or url.query or url.fragment):
        p.error("use an HTTPS URL without credentials, or git@host:path SSH URL")
    for name in ("work_dir", "venv", "checkpoint", "pretrained_backbone", "data_dir",
                 "train_img", "train_ann", "val_img", "val_ann", "archive"):
        value = getattr(a, name)
        if value is not None:
            setattr(a, name, value.expanduser().resolve())
    a.venv = a.venv or a.work_dir / ".venv"
    a.data_dir = a.data_dir or a.work_dir / "data/bdd100k"
    if a.auto_pretrained:
        a.pretrained_backbone = a.work_dir / "checkpoints/fastvit_sa12_imagenet_unfused.pth.tar"
    return a


def command(argv, *, cwd=None, env=None, capture=False, label=None):
    argv = [str(x) for x in argv]
    print("+ " + (label or shlex.join(argv)), flush=True)
    result = subprocess.run(argv, cwd=cwd, env=env, check=True, text=True,
                            stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else None


def clone_source(a, env):
    repo = a.work_dir / "source"
    if repo.exists():
        root = command(["git", "-C", repo, "rev-parse", "--show-toplevel"],
                       env=env, capture=True)
        if Path(root).resolve() != repo.resolve():
            raise ValueError(f"{repo} is not a standalone Git clone")
        origin = command(["git", "-C", repo, "remote", "get-url", "origin"],
                         env=env, capture=True)
        if origin.rstrip("/") != a.repo_url.rstrip("/"):
            raise ValueError("Existing clone has a different origin; use another --work-dir")
        dirty = command(["git", "-C", repo, "status", "--porcelain",
                         "--untracked-files=all"], env=env, capture=True)
        if dirty:
            raise ValueError("Existing clone has local changes; refusing to overwrite them")
    else:
        command(["git", "clone", "--single-branch", "--branch", a.branch,
                 "--", a.repo_url, repo], env=env)
    target = a.commit or a.branch
    command(["git", "-C", repo, "fetch", "origin", target], env=env)
    command(["git", "-C", repo, "checkout", "--detach", "FETCH_HEAD"], env=env)
    sha = command(["git", "-C", repo, "rev-parse", "HEAD"], env=env, capture=True)
    if a.commit and sha.lower() != a.commit.lower():
        raise ValueError("Fetched commit does not match --commit")
    for name in ("train_bdd100k.py", "requirements.txt"):
        if not (repo / name).is_file():
            raise ValueError(f"Clone is missing {name}")
    return repo, sha


def prepare_python(a, repo, env):
    python = a.venv / "bin/python"
    if not python.is_file():
        if a.skip_install:
            raise ValueError("--skip-install requires an existing Linux venv/bin/python")
        command([sys.executable, "-m", "venv", a.venv], env=env)
    if not a.skip_install:
        # Pin an official CUDA wheel pair; requirements.txt must not upgrade it.
        constraints = a.work_dir / "torch-constraints.txt"
        constraints.write_text("torch==2.9.1\ntorchvision==0.24.1\n", encoding="utf-8")
        command([python, "-m", "pip", "install", "--upgrade", "pip"], env=env)
        command([python, "-m", "pip", "install", "torch==2.9.1",
                 "torchvision==0.24.1", "--index-url",
                 "https://download.pytorch.org/whl/cu128"], env=env)
        command([python, "-m", "pip", "install", "-r", repo / "requirements.txt",
                 "-c", constraints], env=env)
    command([python, "-m", "pip", "check"], env=env)
    return python


def probe(python, code, env, *args, label):
    output = command([python, "-c", code, *args], env=env, capture=True, label=label)
    return json.loads(output)


def checkpoint_recipe(a, checkpoint):
    recipe = dict(RECIPE_DEFAULTS)
    if checkpoint is None:
        recipe["model"] = a.model
        return recipe
    saved = checkpoint.get("args")
    metadata = checkpoint.get("metadata")
    if not isinstance(saved, dict) or not isinstance(metadata, dict):
        raise ValueError("Checkpoint needs saved args and metadata for a controlled resume")
    if a.mode == "resume":
        missing = set(RECIPE_DEFAULTS) - set(saved)
        if missing:
            raise ValueError(f"Checkpoint lacks training args: {sorted(missing)}")
        required = {"model_state_dict", "optimizer_state_dict", "scheduler_state_dict"}
        if saved.get("amp", True):
            required.add("scaler_state_dict")
        if not required.issubset(checkpoint.get("state_keys", [])):
            raise ValueError("Checkpoint lacks full model/optimizer/scheduler/scaler state")
        epoch = checkpoint.get("epoch")
        if not isinstance(epoch, int) or epoch < 0 or a.epochs <= epoch + 1:
            raise ValueError("--epochs must be greater than completed checkpoint epochs")
        if saved["scheduler"] == "cosine" and a.epochs != saved.get("epochs"):
            raise ValueError("Changing total epochs changes the cosine schedule; "
                             "use the checkpoint's original --epochs")
        if (saved.get("batch_size"), saved.get("accum_steps", 1)) != (a.batch_size, a.accum_steps):
            print("WARNING: micro-batch/accumulation differ from the checkpoint; "
                  "equal effective batch does not guarantee identical training")
        if (saved["scheduler"] == "step" and saved["lr_steps"]
                and a.epochs > max(saved["lr_steps"]) + 1):
            print(f"NOTE: keeping checkpoint LR milestones {saved['lr_steps']} "
                  f"unchanged for target {a.epochs}; extending epochs does not stretch the schedule")
    recipe.update({k: saved[k] for k in recipe if k in saved})
    arch = metadata.get("architecture", {})
    training = metadata.get("training", {})
    if not isinstance(arch, dict) or not isinstance(training, dict):
        raise ValueError("Invalid checkpoint architecture/training metadata")
    for key, value in {
        "model_name": recipe["model"], "architecture_version": recipe["architecture_version"],
        "fpn_channels": recipe["fpn_channels"], "anchor_sizes": recipe["anchor_sizes"],
        "num_classes": 10, "regression_mode": "box_delta", "reg_max": None,
    }.items():
        if key not in arch or arch[key] != value:
            raise ValueError(f"Checkpoint architecture mismatch/unsupported: {key}")
    for key, value in {
        "classification_loss": recipe["classification_loss"],
        "classification_beta": recipe["focal_gamma"],
        "classification_alpha": recipe["focal_alpha"]
            if recipe["classification_loss"] == "binary_quality_focal" else None,
        "regression_loss": "ciou", "matcher": "fixed_iou_0.4_0.5_force_positive",
    }.items():
        if key not in training or training[key] != value:
            raise ValueError(f"Checkpoint training mismatch/unsupported: {key}")
    return recipe


def data_paths(a):
    return {name: getattr(a, name) or a.data_dir / relative for name, relative in {
        "train_img": "images/100k/train", "val_img": "images/100k/val",
        "train_ann": "annotations/bdd100k_det_train_coco.json",
        "val_ann": "annotations/bdd100k_det_val_coco.json",
    }.items()}


def audit_data(a):
    paths, summary = data_paths(a), {}
    for split in (["val"] if a.mode == "eval" else ["train", "val"]):
        root, annotation = paths[split + "_img"], paths[split + "_ann"]
        if not root.is_dir() or not annotation.is_file():
            raise ValueError(f"Missing {split} data: {root}, {annotation}")
        with annotation.open(encoding="utf-8") as handle:
            data = json.load(handle)
        categories = data.get("categories", [])
        mapping = {c["id"]: ALIASES.get(c["name"], c["name"]) for c in categories}
        if len(categories) != 10 or mapping != dict(enumerate(CLASSES, 1)):
            raise ValueError(f"{annotation}: expected BDD100K IDs 1..10 in canonical order")
        images = data.get("images", [])
        if not images or (split == "train" and len(images) < a.batch_size):
            raise ValueError(f"{split} split is empty or smaller than a training batch")
        # Sample paths, not a costly scan of every image on network storage.
        for item in images[:32]:
            image = (root / item["file_name"]).resolve()
            if not image.is_relative_to(root.resolve()) or not image.is_file():
                raise ValueError(f"Missing or unsafe image path: {image}")
        summary[split] = {"images": len(images), "annotations": len(data.get("annotations", [])),
                          "sampled_paths": min(32, len(images)),
                          "annotation_path": str(annotation)}
        if split == "val" and len(images) != 10000:
            print(f"WARNING: val has {len(images)} images, not the full 10,000-image split.")
    return summary


def training_command(a, python, repo, recipe):
    argv = [str(python), "-u", str(repo / "train_bdd100k.py")]
    for key, value in recipe.items():
        if key != "amp":
            argv.append("--" + key.replace("_", "-"))
            if isinstance(value, list):
                argv.extend(str(x) for x in value)
            else:
                argv.append(str(value))
    argv.append("--amp" if recipe["amp"] else "--no-amp")
    options = {
        "data_dir": a.data_dir, **data_paths(a),
        "epochs": a.epochs, "batch_size": a.batch_size, "accum_steps": a.accum_steps,
        "eval_batch_size": a.eval_batch_size, "workers": a.workers,
        "score_thresh": a.score_thresh, "nms_thresh": a.nms_thresh,
        "max_detections": a.max_detections, "pre_nms_topk": a.pre_nms_topk,
        "eval_interval": 1, "log_interval": 100, "output": a.work_dir / "runs",
    }
    if a.checkpoint:
        options["resume"] = a.checkpoint
    if a.pretrained_backbone:
        options["pretrained_backbone"] = a.pretrained_backbone
    if a.max_eval_samples:
        options["max_eval_samples"] = a.max_eval_samples
    for key, value in options.items():
        argv.extend(["--" + key.replace("_", "-"), str(value)])
    argv.extend(["--save-json", "--no-wandb"])
    if a.mode == "eval":
        argv.append("--eval-only")
    return argv


def stream_job(argv, repo, env, logfile):
    print("+ " + shlex.join(argv), flush=True)
    with logfile.open("w", encoding="utf-8") as handle:
        child = subprocess.Popen(argv, cwd=repo, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in child.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
            return child.wait()
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise
        finally:
            child.stdout.close()


def main(argv=None):
    a = parse_args(argv)
    if a.dry_run:
        print(f"Mode: {a.mode}; initialization: "
              f"{a.pretrained_backbone or a.checkpoint or 'random weights'}")
        print(f"Auto ImageNet download/cache: {a.auto_pretrained}; "
              f"forced redownload: {a.redownload_pretrained}")
        print(f"Clone {a.repo_url}, branch={a.branch}, commit={a.commit or 'branch HEAD'}")
        print(f"Source: {a.work_dir / 'source'}; venv: {a.venv}")
        print("Install pinned torch 2.9.1 + torchvision 0.24.1 CUDA 12.8 wheels"
              if not a.skip_install else "Reuse venv; skip package installation")
        print(f"GPU: {a.gpu}; data preparation requested: "
              f"{bool(a.download_data or a.archive or a.convert_only)}")
        print("Preview only: actual resume recipe is read from the checkpoint at launch.")
        print(shlex.join(training_command(a, a.venv / "bin/python",
                                          a.work_dir / "source", dict(RECIPE_DEFAULTS,
                                                                     model=a.model))))
        return 0
    if sys.platform != "linux" or sys.version_info < (3, 10):
        raise ValueError("Run on Linux with Python >=3.10; --dry-run works on other hosts")
    if a.mode == "train":
        print("New training from epoch 0; fresh optimizer/scheduler/scaler; initialization: "
              f"{a.pretrained_backbone or 'random weights (no pretrained backbone)'}")
    for path in (a.checkpoint, None if a.auto_pretrained else a.pretrained_backbone, a.archive):
        if path is not None and not path.is_file():
            raise ValueError(f"File does not exist on this server: {path}")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, PYTHONUNBUFFERED="1",
               GIT_TERMINAL_PROMPT="0")
    a.work_dir.mkdir(parents=True, exist_ok=True)
    repo, sha = clone_source(a, env)
    python = prepare_python(a, repo, env)
    gpu = probe(python, GPU_PROBE, env, label="CUDA matmul + torchvision NMS preflight")
    print(json.dumps(gpu, indent=2))
    if "H200" not in gpu["name"].upper() and not a.allow_other_gpu:
        raise ValueError("Selected GPU is not H200; use --allow-other-gpu intentionally")
    command([python, repo / "train_bdd100k.py", "--help"], cwd=repo, env=env,
            capture=True, label="Check trainer imports in the cloned source")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    launch = a.work_dir / "launches" / stamp
    launch.mkdir(parents=True)
    pretrained = None
    if a.pretrained_backbone:
        prepared_path = launch / "pretrained_backbone.pth"
        command([python, "-c", PRETRAINED_PREPARE, a.pretrained_backbone, prepared_path,
                 launch / "pretrained.json", "1" if a.auto_pretrained else "0", a.model,
                 "1" if a.redownload_pretrained else "0"],
                cwd=repo, env=env, label="Prepare ImageNet backbone; fresh detection training")
        pretrained = json.loads((launch / "pretrained.json").read_text(encoding="utf-8"))
        a.pretrained_backbone = prepared_path
    checkpoint = None
    if a.checkpoint:
        print("Loading trusted checkpoint (PyTorch pickle); never use an untrusted .pth.")
        checkpoint = probe(python, CHECKPOINT_PROBE, env, str(a.checkpoint),
                           label="Inspect checkpoint on CPU")
    recipe = checkpoint_recipe(a, checkpoint)
    if checkpoint and a.mode == "resume":
        saved = checkpoint["args"]
        old_batch = saved.get("batch_size", 8) * saved.get("accum_steps", 1)
        new_batch = a.batch_size * a.accum_steps
        print(f"Completed epochs: {checkpoint['epoch'] + 1}; target: {a.epochs}")
        if old_batch != new_batch:
            print(f"WARNING: effective batch changes {old_batch} -> {new_batch}; "
                  "LR is NOT automatically scaled. This changes the training protocol.")
    if a.download_data or a.archive or a.convert_only:
        prep = [python, "-m", "tools.download_bdd100k", "--dest-dir", a.data_dir, "--keep-zip"]
        if a.archive:
            prep.extend(["--archive", a.archive])
        if a.convert_only:
            prep.append("--convert-only")
        command(prep, cwd=repo, env=env)
    dataset = audit_data(a)
    if a.max_eval_samples:
        print("WARNING: subset evaluation is a smoke test, not a comparable benchmark.")
    job = training_command(a, python, repo, recipe)
    manifest = {
        "created_utc": stamp, "mode": a.mode, "source_commit": sha,
        "repo_url": a.repo_url, "branch": a.branch, "gpu": gpu,
        "checkpoint_path": str(a.checkpoint) if a.checkpoint else None,
        "checkpoint": checkpoint, "recipe": recipe, "dataset": dataset, "pretrained": pretrained,
        "effective_batch": a.batch_size * a.accum_steps, "command": job,
        "prepare_only": a.prepare_only,
    }
    (launch / "launch.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    freeze = command([python, "-m", "pip", "freeze"], env=env, capture=True)
    (launch / "packages.txt").write_text(freeze + "\n", encoding="utf-8")
    print(f"Launch metadata/logs: {launch}")
    if a.prepare_only:
        print("Preparation complete; no training/evaluation started.")
        return 0
    code = stream_job(job, repo, env, launch / "train.log")
    manifest["exit_code"] = code
    (launch / "launch.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; only completed-epoch checkpoints are resumable.", file=sys.stderr)
        sys.exit(130)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
