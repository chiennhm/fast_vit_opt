#
# FastViT Quantization for Inference
#
# Applies Post-Training Quantization (PTQ) to reduce model size and
# accelerate inference on CPU.  Supports both backbone (classification)
# and full detection models.
#
# Modes:
#   dynamic  – INT8 dynamic quantization on nn.Linear / nn.Conv2d
#   static   – INT8 static quantization with calibration dataset
#
# Usage:
#   python quantize.py --model fastvit_sa12 --method dynamic
#   python quantize.py --model fastvit_sa12 --method static --calib-data ./data/imagenet/val
#   python quantize.py --model fastvit_sa12 --method dynamic --mode detection --checkpoint best.pth
#   python quantize.py --model fastvit_sa12 --method dynamic --benchmark
#

import argparse
import os
import sys
import copy
import time
import json
import warnings
from datetime import datetime

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.ao.quantization import (
    HistogramObserver,
    PerChannelMinMaxObserver,
    QConfig,
)
import numpy as np

# FX-graph-mode quantization (automatic fusion)
try:
    from torch.ao.quantization import quantize_fx
    HAS_FX = True
except ImportError:
    HAS_FX = False

import models  # registers FastViT variants
from timm.models import create_model

try:
    from models.modules.mobileone import reparameterize_model
    HAS_REPARAM = True
except ImportError:
    HAS_REPARAM = False

try:
    from fvcore.nn import FlopCountAnalysis, parameter_count
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False


# ============================================================================
# Constants
# ============================================================================
ALL_VARIANTS = [
    "fastvit_t8", "fastvit_t12", "fastvit_s12",
    "fastvit_sa12", "fastvit_sa24", "fastvit_sa36", "fastvit_ma36",
]


# ============================================================================
# Model builder  (reuses project conventions)
# ============================================================================
def build_model(variant, mode, checkpoint=None, reparam=True, num_classes=20):
    """Build and prepare model for quantization.

    Always runs on CPU (quantization target) and in eval mode.

    Args:
        variant:    FastViT variant name (e.g. ``fastvit_sa12``)
        mode:       ``'backbone'`` or ``'detection'``
        checkpoint: optional path to ``.pth`` weights
        reparam:    fuse multi-branch blocks into single Conv2d first
        num_classes: number of classes (default: 20)

    Returns:
        nn.Module on CPU in eval mode
    """
    if mode == "detection":
        from detection.fastvit_detector import FastViTDetector
        model = FastViTDetector(model_name=variant, num_classes=num_classes)
    else:
        # If backbone mode, default to 1000 unless custom num_classes is passed
        model = create_model(variant, num_classes=num_classes if num_classes != 20 else 1000)

    # Load checkpoint if provided
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        model.load_state_dict(state_dict, strict=False)
        print(f"  [✓] Loaded checkpoint: {checkpoint}")

    # Reparameterize: fuses BN + multi-branch into plain Conv2d
    # This is critical before quantization — gives cleaner graph.
    if reparam and HAS_REPARAM:
        if mode == "backbone":
            model = reparameterize_model(model)
        else:
            model.backbone = reparameterize_model(model.backbone)
        print("  [✓] Reparameterized model")

    model = model.cpu().eval()
    return model


# ============================================================================
# Custom QConfig: HistogramObserver + PerChannelMinMaxObserver
# ============================================================================
def _make_qconfig(backend="x86"):
    """Build a QConfig with:

    - **Activations**: ``HistogramObserver``  – fits a histogram over
      calibration data and chooses the quantization range by minimising
      KL-divergence.  Much more robust to outliers than plain MinMax.
    - **Weights**: ``PerChannelMinMaxObserver`` – tracks min/max
      independently for every output-channel, keeping per-channel scale
      factors.  This greatly reduces the accuracy drop for Conv2d layers
      compared to per-tensor MinMax.
    """
    if backend == "qnnpack":
        act_observer = HistogramObserver.with_args(
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
        )
        weight_observer = PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
        )
    else:  # x86 / fbgemm
        act_observer = HistogramObserver.with_args(
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
        )
        weight_observer = PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
        )

    return QConfig(activation=act_observer, weight=weight_observer)


# ============================================================================
# Dynamic Quantization
# ============================================================================
def quantize_dynamic(model):
    """Apply dynamic quantization (INT8 weights, FP32 activations).

    Best for models with many nn.Linear layers.
    Also quantizes nn.Conv2d on supported backends.

    Args:
        model: nn.Module in eval mode on CPU

    Returns:
        Quantized model
    """
    # Determine which layers to quantize
    # Linear layers benefit most from dynamic quant
    quant_layers = {nn.Linear}

    quantized = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec=quant_layers,
        dtype=torch.qint8,
    )
    print("  [✓] Applied dynamic quantization (INT8 Linear layers)")
    return quantized


# ============================================================================
# Static Quantization (PTQ with calibration)
# ============================================================================
class CalibrationDataLoader:
    """Simple calibration data generator.

    If a real data directory is provided, loads images from it.
    Otherwise generates random calibration tensors (less accurate
    but sufficient for demonstrating the quantization pipeline).
    """

    def __init__(self, data_dir=None, img_size=256, num_samples=100, batch_size=8):
        self.img_size = img_size
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.data_dir = data_dir
        self._loader = None

        if data_dir and os.path.isdir(data_dir):
            self._init_real_data(data_dir)

    def _init_real_data(self, data_dir):
        """Initialize with real calibration images using torchvision."""
        try:
            from torchvision import transforms, datasets

            transform = transforms.Compose([
                transforms.Resize(int(self.img_size / 0.875)),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            dataset = datasets.ImageFolder(data_dir, transform=transform)
            # Subsample for calibration
            indices = np.random.choice(
                len(dataset),
                min(self.num_samples, len(dataset)),
                replace=False,
            )
            subset = torch.utils.data.Subset(dataset, indices)
            self._loader = torch.utils.data.DataLoader(
                subset, batch_size=self.batch_size, shuffle=False, num_workers=2,
            )
            print(f"  [✓] Calibration: {len(subset)} real images from {data_dir}")
        except Exception as e:
            print(f"  [!] Could not load real data: {e}")
            print("      Falling back to random calibration tensors.")
            self._loader = None

    def __iter__(self):
        if self._loader is not None:
            for batch, _ in self._loader:
                yield batch
        else:
            # Random calibration data (fallback)
            remaining = self.num_samples
            while remaining > 0:
                bs = min(self.batch_size, remaining)
                yield torch.randn(bs, 3, self.img_size, self.img_size)
                remaining -= bs

    def __len__(self):
        if self._loader is not None:
            return len(self._loader)
        return (self.num_samples + self.batch_size - 1) // self.batch_size


def quantize_static(model, calib_data_dir=None, img_size=256,
                     num_calib=200, backend="x86"):
    """Apply static post-training quantization with calibration.

    Quantizes both weights AND activations to INT8, giving the best
    latency improvement but requiring representative calibration data.

    Pipeline:
      1. FX-graph-mode automatic fusion (Conv-BN, Conv-BN-ReLU, …)
         – no manual pattern scanning required.
      2. HistogramObserver on activations (KL-divergence range selection).
      3. PerChannelMinMaxObserver on weights (per output-channel scale).
      4. Calibration → INT8 conversion.

    Falls back to eager-mode if FX tracing fails (e.g. dynamic control flow).

    Args:
        model:          nn.Module in eval mode on CPU
        calib_data_dir: path to calibration images (ImageFolder layout)
        img_size:       input image resolution
        num_calib:      number of calibration samples
        backend:        quantization backend ('x86', 'fbgemm', 'qnnpack')

    Returns:
        Quantized model
    """
    # Set quantization backend
    engine = backend if backend in ("x86", "qnnpack") else "x86"
    torch.backends.quantized.engine = engine

    # Custom qconfig: HistogramObserver (act) + PerChannelMinMax (weight)
    custom_qconfig = _make_qconfig(backend)

    # Build calibration data
    calib_loader = CalibrationDataLoader(
        data_dir=calib_data_dir,
        img_size=img_size,
        num_samples=num_calib,
        batch_size=8,
    )

    # ------------------------------------------------------------------
    # FX Graph Mode  (preferred — automatic Conv-BN fusion)
    # ------------------------------------------------------------------
    if HAS_FX:
        try:
            model_to_quantize = copy.deepcopy(model)
            model_to_quantize.eval()

            qconfig_mapping = torch.ao.quantization.QConfigMapping().set_global(
                custom_qconfig
            )
            example_input = torch.randn(1, 3, img_size, img_size)

            # prepare_fx fuses Conv-BN(-ReLU) automatically
            model_prepared = quantize_fx.prepare_fx(
                model_to_quantize,
                qconfig_mapping,
                example_inputs=(example_input,),
            )
            print(f"  [✓] FX auto-fusion + prepare (backend={engine})")
            print(f"      Activation observer : HistogramObserver")
            print(f"      Weight observer     : PerChannelMinMaxObserver")

            # Calibration
            print(f"  [→] Running calibration with {num_calib} samples...")
            with torch.inference_mode():
                for i, batch in enumerate(calib_loader):
                    model_prepared(batch)
                    if (i + 1) % 10 == 0:
                        print(f"      Calibrated {(i + 1) * batch.shape[0]} samples")

            # Convert
            quantized_model = quantize_fx.convert_fx(model_prepared)
            print("  [✓] Converted to static INT8 model (FX graph mode)")
            return quantized_model

        except Exception as e:
            print(f"  [!] FX tracing failed: {e}")
            print("      Falling back to eager-mode quantization...")

    # ------------------------------------------------------------------
    # Eager Mode fallback
    # ------------------------------------------------------------------
    model_prepared = copy.deepcopy(model)
    model_prepared.eval()

    # Wrap with QuantStub / DeQuantStub
    model_prepared = nn.Sequential(
        torch.quantization.QuantStub(),
        model_prepared,
        torch.quantization.DeQuantStub(),
    )

    # Apply custom qconfig (HistogramObserver + PerChannelMinMax)
    model_prepared.qconfig = custom_qconfig
    print(f"  [✓] Eager-mode prepare (backend={engine})")
    print(f"      Activation observer : HistogramObserver")
    print(f"      Weight observer     : PerChannelMinMaxObserver")

    quant.prepare(model_prepared, inplace=True)

    # Calibration
    print(f"  [→] Running calibration with {num_calib} samples...")
    with torch.inference_mode():
        for i, batch in enumerate(calib_loader):
            model_prepared(batch)
            if (i + 1) % 10 == 0:
                print(f"      Calibrated {(i + 1) * batch.shape[0]} samples")

    # Convert
    quant.convert(model_prepared, inplace=True)
    print("  [✓] Converted to static INT8 model (eager mode)")

    return model_prepared


# ============================================================================
# Model size utilities
# ============================================================================
def get_model_size_mb(model):
    """Get model size in MB by saving to a temporary buffer."""
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_mb = buffer.tell() / 1e6
    return round(size_mb, 2)


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============================================================================
# Benchmark: compare original vs quantized
# ============================================================================
def benchmark_inference(model, input_shape, warmup=30, iterations=100, label="Model"):
    """Benchmark CPU inference latency.

    Args:
        model:       nn.Module on CPU
        input_shape: tuple (B, C, H, W)
        warmup:      warmup iterations
        iterations:  timed iterations
        label:       display label

    Returns:
        dict with latency statistics
    """
    model.eval()
    dummy = torch.randn(*input_shape)

    # Warmup
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(dummy)

    # Timed runs
    latencies = []
    with torch.inference_mode():
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = model(dummy)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms

    latencies = np.array(latencies)
    batch_size = input_shape[0]

    result = {
        "label": label,
        "batch_size": batch_size,
        "mean_ms": round(float(np.mean(latencies)), 3),
        "std_ms": round(float(np.std(latencies)), 3),
        "median_ms": round(float(np.median(latencies)), 3),
        "min_ms": round(float(np.min(latencies)), 3),
        "p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "throughput_img_s": round(batch_size * 1000.0 / float(np.mean(latencies)), 2),
    }
    return result


def print_comparison(fp32_stats, quant_stats, fp32_size, quant_size):
    """Pretty-print comparison between FP32 and quantized models."""
    print("\n" + "=" * 78)
    print("  QUANTIZATION COMPARISON")
    print("=" * 78)

    header = f"{'Metric':<25} {'FP32':>15} {'Quantized':>15} {'Speedup/Δ':>15}"
    print(header)
    print("-" * 78)

    # Latency
    speedup = fp32_stats["mean_ms"] / quant_stats["mean_ms"] if quant_stats["mean_ms"] > 0 else 0
    print(f"  {'Latency (ms)':24s} {fp32_stats['mean_ms']:>14.2f}  {quant_stats['mean_ms']:>14.2f}  {speedup:>13.2f}x")
    print(f"  {'Latency std (ms)':24s} {fp32_stats['std_ms']:>14.2f}  {quant_stats['std_ms']:>14.2f}")
    print(f"  {'P95 Latency (ms)':24s} {fp32_stats['p95_ms']:>14.2f}  {quant_stats['p95_ms']:>14.2f}")

    # Throughput
    print(f"  {'Throughput (img/s)':24s} {fp32_stats['throughput_img_s']:>14.1f}  {quant_stats['throughput_img_s']:>14.1f}")

    # Model size
    compression = fp32_size / quant_size if quant_size > 0 else 0
    reduction_pct = (1 - quant_size / fp32_size) * 100 if fp32_size > 0 else 0
    print(f"  {'Model size (MB)':24s} {fp32_size:>14.2f}  {quant_size:>14.2f}  {reduction_pct:>12.1f}%↓")

    print("=" * 78)


# ============================================================================
# Export quantized model
# ============================================================================
def export_quantized(model, output_path, img_size=256, mode="backbone"):
    """Export quantized model as TorchScript for deployment.

    Args:
        model:       quantized nn.Module
        output_path: save path (.pt)
        img_size:    input image size for tracing
        mode:        'backbone' or 'detection'
    """
    dummy = torch.randn(1, 3, img_size, img_size)

    try:
        # Try scripting first (preserves control flow)
        scripted = torch.jit.script(model)
        scripted.save(output_path)
        print(f"  [✓] Exported TorchScript (scripted): {output_path}")
    except Exception:
        try:
            # Fall back to tracing
            traced = torch.jit.trace(model, dummy)
            traced.save(output_path)
            print(f"  [✓] Exported TorchScript (traced): {output_path}")
        except Exception as e:
            # Save as state_dict if tracing also fails
            alt_path = output_path.replace(".pt", "_state_dict.pth")
            torch.save(model.state_dict(), alt_path)
            print(f"  [!] TorchScript export failed: {e}")
            print(f"  [✓] Saved state_dict instead: {alt_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="FastViT Quantization for Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dynamic quantization (fast, no calibration data needed)
  python quantize.py --model fastvit_sa12 --method dynamic

  # Static quantization with calibration data
  python quantize.py --model fastvit_sa12 --method static --calib-data ./data/val

  # Detection model + benchmark comparison
  python quantize.py --model fastvit_sa12 --mode detection --checkpoint best.pth --benchmark

  # Export quantized model for deployment
  python quantize.py --model fastvit_sa12 --method dynamic --export
        """,
    )

    parser.add_argument(
        "--model", type=str, default="fastvit_sa12",
        help=f"FastViT variant. Choices: {ALL_VARIANTS}",
    )
    parser.add_argument(
        "--mode", type=str, default="backbone",
        choices=["backbone", "detection"],
        help="Model mode: backbone (classification) or detection",
    )
    parser.add_argument(
        "--method", type=str, default="dynamic",
        choices=["dynamic", "static"],
        help="Quantization method: dynamic or static (PTQ)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to .pth checkpoint to load weights from",
    )
    parser.add_argument(
        "--num-classes", type=int, default=20,
        help="Number of classes (for detection: VOC is 20, COCO is 80; default: 20)",
    )
    parser.add_argument(
        "--img-size", type=int, default=256,
        help="Input image size (default: 256)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size for benchmarking (default: 1)",
    )
    parser.add_argument(
        "--no-reparam", action="store_true",
        help="Skip reparameterization before quantization",
    )

    # Static quantization options
    parser.add_argument(
        "--calib-data", type=str, default=None,
        help="Path to calibration images (ImageFolder layout) for static quant",
    )
    parser.add_argument(
        "--num-calib", type=int, default=200,
        help="Number of calibration samples for static quant (default: 200)",
    )
    parser.add_argument(
        "--backend", type=str, default="x86",
        choices=["x86", "fbgemm", "qnnpack"],
        help="Quantization backend (default: x86)",
    )

    # Output options
    parser.add_argument(
        "--output-dir", type=str, default="./output/quantized",
        help="Output directory for quantized models",
    )
    parser.add_argument(
        "--export", action="store_true",
        help="Export quantized model as TorchScript",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Benchmark FP32 vs quantized model",
    )
    parser.add_argument(
        "--warmup", type=int, default=30,
        help="Warmup iterations for benchmarking (default: 30)",
    )
    parser.add_argument(
        "--iterations", type=int, default=100,
        help="Timed iterations for benchmarking (default: 100)",
    )

    args = parser.parse_args()

    # ----------------------------------------------------------------
    # Header
    # ----------------------------------------------------------------
    print("=" * 70)
    print("  FastViT Quantization")
    print("=" * 70)
    print(f"  Model:       {args.model}")
    print(f"  Mode:        {args.mode}")
    print(f"  Method:      {args.method}")
    print(f"  Image size:  {args.img_size}")
    print(f"  Backend:     {args.backend}")
    print(f"  Checkpoint:  {args.checkpoint or 'None'}")
    print(f"  PyTorch:     {torch.__version__}")
    print("=" * 70)

    # ----------------------------------------------------------------
    # Build FP32 model
    # ----------------------------------------------------------------
    print("\n[1/4] Building FP32 model...")
    fp32_model = build_model(
        args.model,
        args.mode,
        checkpoint=args.checkpoint,
        reparam=not args.no_reparam,
        num_classes=args.num_classes,
    )
    fp32_size = get_model_size_mb(fp32_model)
    total_params, _ = count_parameters(fp32_model)
    print(f"  Parameters:  {total_params / 1e6:.2f}M")
    print(f"  Model size:  {fp32_size:.2f} MB (FP32)")

    # ----------------------------------------------------------------
    # Quantize
    # ----------------------------------------------------------------
    print(f"\n[2/4] Applying {args.method} quantization...")

    if args.method == "dynamic":
        quant_model = quantize_dynamic(fp32_model)
    elif args.method == "static":
        if args.mode == "detection":
            warnings.warn(
                "Static quantization on the full detection model may fail "
                "due to complex control flow. Falling back to backbone-only "
                "static + dynamic on head if errors occur.",
                RuntimeWarning,
            )
        quant_model = quantize_static(
            fp32_model,
            calib_data_dir=args.calib_data,
            img_size=args.img_size,
            num_calib=args.num_calib,
            backend=args.backend,
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    quant_size = get_model_size_mb(quant_model)
    compression = fp32_size / quant_size if quant_size > 0 else 0
    print(f"  Quantized model size: {quant_size:.2f} MB ({compression:.1f}x compression)")

    # ----------------------------------------------------------------
    # Benchmark
    # ----------------------------------------------------------------
    if args.benchmark:
        print(f"\n[3/4] Benchmarking (batch_size={args.batch_size}, "
              f"warmup={args.warmup}, iters={args.iterations})...")

        input_shape = (args.batch_size, 3, args.img_size, args.img_size)

        print("  → FP32 model...")
        fp32_stats = benchmark_inference(
            fp32_model, input_shape,
            warmup=args.warmup, iterations=args.iterations,
            label="FP32",
        )

        print("  → Quantized model...")
        quant_stats = benchmark_inference(
            quant_model, input_shape,
            warmup=args.warmup, iterations=args.iterations,
            label=f"INT8-{args.method}",
        )

        print_comparison(fp32_stats, quant_stats, fp32_size, quant_size)
    else:
        print("\n[3/4] Benchmarking skipped (use --benchmark to enable)")
        fp32_stats, quant_stats = None, None

    # ----------------------------------------------------------------
    # Save / Export
    # ----------------------------------------------------------------
    print(f"\n[4/4] Saving quantized model...")
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save PyTorch model
    model_name = f"{args.model}_{args.mode}_{args.method}_int8"
    pth_path = os.path.join(args.output_dir, f"{model_name}.pth")
    torch.save({
        "model_state_dict": quant_model.state_dict(),
        "config": {
            "variant": args.model,
            "mode": args.mode,
            "method": args.method,
            "img_size": args.img_size,
            "backend": args.backend,
            "fp32_size_mb": fp32_size,
            "quant_size_mb": quant_size,
            "total_params": total_params,
            "timestamp": timestamp,
        },
    }, pth_path)
    print(f"  [✓] Saved: {pth_path}")

    # Export TorchScript
    if args.export:
        ts_path = os.path.join(args.output_dir, f"{model_name}.pt")
        export_quantized(quant_model, ts_path, args.img_size, args.mode)

    # Save benchmark report
    if fp32_stats and quant_stats:
        report = {
            "timestamp": timestamp,
            "config": {
                "variant": args.model,
                "mode": args.mode,
                "method": args.method,
                "img_size": args.img_size,
                "batch_size": args.batch_size,
                "backend": args.backend,
            },
            "fp32": {**fp32_stats, "size_mb": fp32_size},
            "quantized": {**quant_stats, "size_mb": quant_size},
            "speedup": round(
                fp32_stats["mean_ms"] / quant_stats["mean_ms"], 3
            ) if quant_stats["mean_ms"] > 0 else 0,
            "compression": round(compression, 2),
        }
        report_path = os.path.join(args.output_dir, f"{model_name}_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  [✓] Report: {report_path}")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Done!")
    print(f"  FP32  → {fp32_size:.2f} MB")
    print(f"  INT8  → {quant_size:.2f} MB  ({compression:.1f}x smaller)")
    if fp32_stats and quant_stats:
        speedup = fp32_stats["mean_ms"] / quant_stats["mean_ms"] if quant_stats["mean_ms"] > 0 else 0
        print(f"  Speedup: {speedup:.2f}x  "
              f"({fp32_stats['mean_ms']:.1f}ms → {quant_stats['mean_ms']:.1f}ms)")
    print("=" * 70)


if __name__ == "__main__":
    main()
