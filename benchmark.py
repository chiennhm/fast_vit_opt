#
# FastViT Benchmark: Throughput, Latency, FLOPs, Memory, Energy
#
# Usage:
#   python benchmark.py --model fastvit_sa12                          # backbone only
#   python benchmark.py --model fastvit_sa12 --mode detection         # full detector
#   python benchmark.py --model all --mode backbone                   # compare all variants
#   python benchmark.py --model fastvit_t8 --img-size 256 512 --batch-size 1 8 16
#

import argparse
import os
import sys
import time
import json
import csv
import platform
import statistics
from datetime import datetime
from contextlib import contextmanager

import torch
import torch.nn as nn
import numpy as np

import models  # registers FastViT variants
from timm.models import create_model

# Optional imports
try:
    from fvcore.nn import FlopCountAnalysis, parameter_count
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

try:
    from models.modules.mobileone import reparameterize_model
    HAS_REPARAM = True
except ImportError:
    HAS_REPARAM = False


# ============================================================================
# Constants
# ============================================================================
ALL_VARIANTS = [
    "fastvit_t8", "fastvit_t12", "fastvit_s12",
    "fastvit_sa12", "fastvit_sa24", "fastvit_sa36", "fastvit_ma36",
]


# ============================================================================
# GPU Energy Monitor (nvidia-smi via pynvml)
# ============================================================================
class GPUEnergyMonitor:
    """Monitor GPU power draw via NVML to estimate energy consumption."""

    def __init__(self, device_index=0, sample_interval_ms=50):
        self.device_index = device_index
        self.sample_interval = sample_interval_ms / 1000.0
        self.samples = []
        self._handle = None
        self._available = False

        if HAS_PYNVML and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                self._available = True
            except Exception:
                pass

    @property
    def available(self):
        return self._available

    def read_power_watts(self):
        """Read current GPU power in Watts."""
        if not self._available:
            return 0.0
        try:
            mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
            return mw / 1000.0
        except Exception:
            return 0.0

    def start(self):
        self.samples = []

    def sample(self):
        if self._available:
            self.samples.append((time.perf_counter(), self.read_power_watts()))

    def compute_energy_joules(self):
        """Compute total energy (Joules) via trapezoidal integration."""
        if len(self.samples) < 2:
            return 0.0
        energy = 0.0
        for i in range(1, len(self.samples)):
            dt = self.samples[i][0] - self.samples[i - 1][0]
            avg_power = (self.samples[i][1] + self.samples[i - 1][1]) / 2.0
            energy += avg_power * dt
        return energy

    def summary(self):
        if len(self.samples) < 2:
            return {"available": False}
        powers = [s[1] for s in self.samples]
        total_time = self.samples[-1][0] - self.samples[0][0]
        return {
            "available": True,
            "total_energy_J": round(self.compute_energy_joules(), 3),
            "avg_power_W": round(statistics.mean(powers), 2),
            "peak_power_W": round(max(powers), 2),
            "duration_s": round(total_time, 3),
            "num_samples": len(self.samples),
        }


# ============================================================================
# Memory tracker
# ============================================================================
@contextmanager
def track_gpu_memory(device):
    """Context manager that tracks peak GPU memory allocation."""
    if device.type != "cuda":
        yield {"peak_mb": 0, "allocated_mb": 0}
        return

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    mem_before = torch.cuda.memory_allocated(device)

    result = {}
    yield result

    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    current = torch.cuda.memory_allocated(device)
    result["peak_mb"] = round(peak / 1e6, 2)
    result["allocated_mb"] = round((current - mem_before) / 1e6, 2)


# ============================================================================
# FLOPs / Params
# ============================================================================
def count_flops_params(model, input_tensor):
    """Count FLOPs and parameters using fvcore."""
    if not HAS_FVCORE:
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            "total_params_M": round(total_params / 1e6, 2),
            "trainable_params_M": round(trainable / 1e6, 2),
            "flops_G": None,
            "note": "Install fvcore for FLOPs: pip install fvcore",
        }

    flops = FlopCountAnalysis(model, input_tensor)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    params = parameter_count(model)

    return {
        "total_params_M": round(params[""] / 1e6, 2),
        "flops_G": round(flops.total() / 1e9, 2),
    }


# ============================================================================
# Latency & Throughput benchmark
# ============================================================================
def benchmark_latency(
    model, input_shape, device, warmup=50, iterations=200, use_amp=False,
    energy_monitor=None,
):
    """Measure inference latency and throughput.

    Returns dict with latency stats (ms) and throughput (img/s).
    """
    model.eval()
    dummy = torch.randn(*input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    _ = model(dummy)
            else:
                _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Energy monitoring
    if energy_monitor and energy_monitor.available:
        energy_monitor.start()

    # Timed runs
    latencies = []
    batch_size = input_shape[0]

    for _ in range(iterations):
        if energy_monitor and energy_monitor.available:
            energy_monitor.sample()

        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            t0 = time.perf_counter()

        with torch.no_grad():
            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    _ = model(dummy)
            else:
                _ = model(dummy)

        if device.type == "cuda":
            end_event.record()
            torch.cuda.synchronize(device)
            latencies.append(start_event.elapsed_time(end_event))
        else:
            latencies.append((time.perf_counter() - t0) * 1000)

        if energy_monitor and energy_monitor.available:
            energy_monitor.sample()

    latencies = np.array(latencies)

    result = {
        "batch_size": batch_size,
        "iterations": iterations,
        "mean_ms": round(float(np.mean(latencies)), 3),
        "std_ms": round(float(np.std(latencies)), 3),
        "median_ms": round(float(np.median(latencies)), 3),
        "min_ms": round(float(np.min(latencies)), 3),
        "max_ms": round(float(np.max(latencies)), 3),
        "p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "p99_ms": round(float(np.percentile(latencies, 99)), 3),
        "throughput_img_per_s": round(batch_size * 1000.0 / float(np.mean(latencies)), 2),
    }

    if energy_monitor and energy_monitor.available:
        e = energy_monitor.summary()
        result["energy"] = e
        if e["available"] and result["throughput_img_per_s"] > 0:
            total_images = batch_size * iterations
            result["energy_per_image_mJ"] = round(
                e["total_energy_J"] / total_images * 1000, 3
            )

    return result


# ============================================================================
# Build model helper
# ============================================================================
def build_model(variant, mode, device, reparam=False):
    """Build backbone or full detector model."""
    if mode == "detection":
        from detection.fastvit_detector import FastViTDetector
        model = FastViTDetector(model_name=variant, num_classes=20)
    else:
        model = create_model(variant, num_classes=1000)

    if reparam and HAS_REPARAM and mode == "backbone":
        model = reparameterize_model(model)

    model = model.to(device).eval()
    return model


# ============================================================================
# System info
# ============================================================================
def get_system_info(device):
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(device)
        props = torch.cuda.get_device_properties(device)
        info["gpu_memory_GB"] = round(props.total_mem / 1e9, 2)
        info["cuda_version"] = torch.version.cuda or "N/A"
        info["cudnn_version"] = str(torch.backends.cudnn.version())
    return info


# ============================================================================
# Pretty print
# ============================================================================
def print_results(results, system_info):
    """Print benchmark results as a formatted table."""
    print("\n" + "=" * 90)
    print("SYSTEM INFO")
    print("=" * 90)
    for k, v in system_info.items():
        print(f"  {k:20s}: {v}")

    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS")
    print("=" * 90)

    # Header
    header = (
        f"{'Model':<16} {'Mode':<10} {'ImgSize':>7} {'BS':>4} "
        f"{'Params(M)':>9} {'GFLOPs':>7} "
        f"{'Latency(ms)':>11} {'±std':>6} {'P95(ms)':>8} "
        f"{'Tput(img/s)':>11} {'PeakMem(MB)':>11}"
    )
    energy_col = any("energy" in r and r["energy"].get("available") for r in results)
    if energy_col:
        header += f" {'Energy(mJ/img)':>14} {'AvgPwr(W)':>10}"
    print(header)
    print("-" * len(header))

    for r in results:
        flops_str = f"{r['flops_G']:.1f}" if r.get("flops_G") else "N/A"
        line = (
            f"{r['variant']:<16} {r['mode']:<10} {r['img_size']:>7} {r['batch_size']:>4} "
            f"{r['params_M']:>9.2f} {flops_str:>7} "
            f"{r['mean_ms']:>11.2f} {r['std_ms']:>6.2f} {r['p95_ms']:>8.2f} "
            f"{r['throughput']:>11.1f} {r['peak_mem_mb']:>11.1f}"
        )
        if energy_col:
            e_str = f"{r.get('energy_per_image_mJ', 'N/A')}"
            p_str = f"{r.get('avg_power_W', 'N/A')}"
            line += f" {e_str:>14} {p_str:>10}"
        print(line)

    print("=" * len(header))


def save_results(results, system_info, output_dir):
    """Save results to JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    data = {"system": system_info, "results": results, "timestamp": timestamp}
    json_path = os.path.join(output_dir, f"benchmark_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    # CSV
    csv_path = os.path.join(output_dir, f"benchmark_{timestamp}.csv")
    if results:
        keys = results[0].keys()
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in results:
                w.writerow({k: str(v) if isinstance(v, dict) else v for k, v in r.items()})

    print(f"\nResults saved to: {json_path}")
    print(f"                  {csv_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="FastViT Benchmark")
    parser.add_argument(
        "--model", type=str, nargs="+", default=["fastvit_sa12"],
        help="Model variants to benchmark. Use 'all' for all variants.",
    )
    parser.add_argument(
        "--mode", type=str, default="backbone", choices=["backbone", "detection"],
        help="Benchmark mode: backbone (classification) or detection (full detector)",
    )
    parser.add_argument(
        "--img-size", type=int, nargs="+", default=[256],
        help="Input image sizes to test (default: 256)",
    )
    parser.add_argument(
        "--batch-size", type=int, nargs="+", default=[1],
        help="Batch sizes to test (default: 1)",
    )
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=200, help="Timed iterations")
    parser.add_argument("--amp", action="store_true", help="Use FP16 (AMP)")
    parser.add_argument("--reparam", action="store_true", help="Reparameterize model before benchmarking")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--energy", action="store_true", help="Enable GPU energy monitoring (requires pynvml)")
    parser.add_argument("--output", type=str, default="./output/benchmark", help="Output directory")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to files")

    args = parser.parse_args()

    # Resolve models
    variants = ALL_VARIANTS if "all" in args.model else args.model

    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda:0")
    system_info = get_system_info(device)

    print("=" * 70)
    print("FastViT Benchmark")
    print("=" * 70)
    print(f"  Device:      {device} ({system_info.get('gpu_name', 'CPU')})")
    print(f"  Models:      {variants}")
    print(f"  Image sizes: {args.img_size}")
    print(f"  Batch sizes: {args.batch_size}")
    print(f"  AMP:         {args.amp}")
    print(f"  Reparam:     {args.reparam}")
    print(f"  Iterations:  {args.iterations} (warmup: {args.warmup})")
    print("=" * 70)

    energy_monitor = None
    if args.energy:
        energy_monitor = GPUEnergyMonitor()
        if energy_monitor.available:
            print(f"  Energy:      Enabled (pynvml)")
        else:
            print(f"  Energy:      Not available (install pynvml or check GPU)")

    all_results = []

    for variant in variants:
        for img_size in args.img_size:
            for batch_size in args.batch_size:
                tag = f"{variant} | {args.mode} | {img_size}px | bs={batch_size}"
                print(f"\n>>> {tag}")

                try:
                    model = build_model(variant, args.mode, device, reparam=args.reparam)

                    # FLOPs / params (always batch=1)
                    flop_input = torch.randn(1, 3, img_size, img_size, device=device)
                    fp = count_flops_params(model, flop_input)

                    # Memory tracking
                    input_shape = (batch_size, 3, img_size, img_size)
                    with track_gpu_memory(device) as mem:
                        lat = benchmark_latency(
                            model, input_shape, device,
                            warmup=args.warmup, iterations=args.iterations,
                            use_amp=args.amp, energy_monitor=energy_monitor,
                        )

                    row = {
                        "variant": variant,
                        "mode": args.mode,
                        "img_size": img_size,
                        "batch_size": batch_size,
                        "amp": args.amp,
                        "reparam": args.reparam,
                        "params_M": fp["total_params_M"],
                        "flops_G": fp.get("flops_G"),
                        "mean_ms": lat["mean_ms"],
                        "std_ms": lat["std_ms"],
                        "median_ms": lat["median_ms"],
                        "min_ms": lat["min_ms"],
                        "p95_ms": lat["p95_ms"],
                        "p99_ms": lat["p99_ms"],
                        "throughput": lat["throughput_img_per_s"],
                        "peak_mem_mb": mem.get("peak_mb", 0),
                    }

                    # Energy
                    if "energy" in lat:
                        row["energy"] = lat["energy"]
                        row["energy_per_image_mJ"] = lat.get("energy_per_image_mJ")
                        row["avg_power_W"] = lat["energy"].get("avg_power_W")

                    all_results.append(row)

                    print(
                        f"    Params: {fp['total_params_M']:.2f}M | "
                        f"FLOPs: {fp.get('flops_G', 'N/A')}G | "
                        f"Latency: {lat['mean_ms']:.2f}±{lat['std_ms']:.2f}ms | "
                        f"Throughput: {lat['throughput_img_per_s']:.1f} img/s | "
                        f"PeakMem: {mem.get('peak_mb', 0):.1f}MB"
                    )
                    if "energy_per_image_mJ" in row and row["energy_per_image_mJ"]:
                        print(f"    Energy: {row['energy_per_image_mJ']:.2f} mJ/img | "
                              f"Avg power: {row['avg_power_W']:.1f}W")

                except Exception as e:
                    print(f"    ERROR: {e}")
                    continue
                finally:
                    # Free GPU memory
                    if device.type == "cuda":
                        del model
                        torch.cuda.empty_cache()

    # Print summary table
    if all_results:
        print_results(all_results, system_info)
        if not args.no_save:
            save_results(all_results, system_info, args.output)


if __name__ == "__main__":
    main()
