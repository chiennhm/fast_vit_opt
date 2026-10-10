"""Utilities for inspecting intermediate detector activations on one image."""

from __future__ import annotations

import csv
import html
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FPN_LEVELS = ("P2", "P3", "P4", "P5", "P6")


def _resampling_bilinear():
    return getattr(Image, "Resampling", Image).BILINEAR


def prepare_single_image(
    image_path: str | Path,
    short_side: int = 800,
    max_size: int = 1333,
    padding_multiple: int = 32,
) -> Tuple[torch.Tensor, Image.Image, Image.Image, Dict[str, object]]:
    """Apply the detector's validation preprocessing to one RGB image."""
    if short_side < 1 or max_size < 1 or padding_multiple < 1:
        raise ValueError("Image sizes and padding_multiple must be positive")

    original = Image.open(image_path).convert("RGB")
    orig_w, orig_h = original.size
    scale = short_side / min(orig_w, orig_h)
    if scale * max(orig_w, orig_h) > max_size:
        scale = max_size / max(orig_w, orig_h)

    resized_w = max(1, int(round(orig_w * scale)))
    resized_h = max(1, int(round(orig_h * scale)))
    resized = original.resize((resized_w, resized_h), _resampling_bilinear())

    padded_w = math.ceil(resized_w / padding_multiple) * padding_multiple
    padded_h = math.ceil(resized_h / padding_multiple) * padding_multiple
    fill = tuple(int(round(value * 255)) for value in IMAGENET_MEAN)
    padded = Image.new("RGB", (padded_w, padded_h), fill)
    padded.paste(resized, (0, 0))

    array = np.asarray(padded, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = ((tensor - mean) / std).unsqueeze(0)

    metadata = {
        "path": str(Path(image_path).resolve()),
        "original_size": [orig_h, orig_w],
        "resized_size": [resized_h, resized_w],
        "padded_size": [padded_h, padded_w],
        "scale_x": resized_w / orig_w,
        "scale_y": resized_h / orig_h,
        "normalization": "imagenet",
    }
    return tensor, original, padded, metadata


def iter_output_tensors(value, prefix="output") -> Iterable[Tuple[str, torch.Tensor]]:
    """Yield every tensor nested in a module output."""
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from iter_output_tensors(item, f"{prefix}.{index}")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from iter_output_tensors(item, f"{prefix}.{key}")


def tensor_statistics(tensor: torch.Tensor) -> Dict[str, object]:
    """Return scalar diagnostics without retaining the source tensor."""
    # Keep reductions on the source device. In --stats-only mode this avoids
    # copying every full activation from GPU to CPU.
    data = tensor.detach().float()
    finite = torch.isfinite(data)
    finite_fraction = float(finite.float().mean()) if data.numel() else 1.0
    values = data[finite]

    if values.numel():
        minimum = float(values.min())
        maximum = float(values.max())
        mean = float(values.mean())
        std = float(values.std(unbiased=False))
        zero_fraction = float((values.abs() < 1e-8).float().mean())
    else:
        minimum = maximum = mean = std = zero_fraction = float("nan")

    dead_channel_fraction = None
    if data.ndim == 4 and data.shape[1] > 0:
        clean = torch.nan_to_num(data[0], nan=0.0, posinf=0.0, neginf=0.0)
        channel_std = clean.flatten(1).std(dim=1, unbiased=False)
        dead_channel_fraction = float((channel_std < 1e-8).float().mean())

    flags = []
    if finite_fraction < 1.0:
        flags.append("nan_or_inf")
    if values.numel() and std < 1e-8:
        flags.append("constant")
    if values.numel() and zero_fraction > 0.99:
        flags.append("mostly_zero")
    if values.numel() and max(abs(minimum), abs(maximum)) > 1e4:
        flags.append("large_magnitude")
    if dead_channel_fraction is not None and dead_channel_fraction > 0.5:
        flags.append("many_dead_channels")

    return {
        "shape": list(data.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "numel": data.numel(),
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "std": std,
        "zero_fraction": zero_fraction,
        "finite_fraction": finite_fraction,
        "dead_channel_fraction": dead_channel_fraction,
        "flags": flags,
    }


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "tensor"


def _normalize_map(array: np.ndarray, fixed_probability_range: bool = False) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=np.float32)
    clean = np.where(finite, array, 0.0)
    if fixed_probability_range:
        return np.clip(clean, 0.0, 1.0)
    values = clean[finite]
    low, high = np.percentile(values, [1.0, 99.0])
    if high <= low + 1e-12:
        low, high = float(values.min()), float(values.max())
    if high <= low + 1e-12:
        return np.zeros_like(clean, dtype=np.float32)
    return np.clip((clean - low) / (high - low), 0.0, 1.0)


def _colorize(normalized: np.ndarray) -> Image.Image:
    """Small dependency-free blue/cyan/yellow/red colour map."""
    value = np.clip(normalized, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * value - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * value - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * value - 1.0), 0.0, 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")


def _map_panel(
    values: np.ndarray,
    base_image: Image.Image,
    title: str,
    fixed_probability_range: bool = False,
) -> Image.Image:
    normalized = _normalize_map(values, fixed_probability_range)
    gray = Image.fromarray((normalized * 255).astype(np.uint8), mode="L")
    gray = gray.resize(base_image.size, _resampling_bilinear())
    normalized_resized = np.asarray(gray, dtype=np.float32) / 255.0
    heatmap = _colorize(normalized_resized)
    overlay = Image.blend(base_image.convert("RGB"), heatmap, 0.5)

    header = 30
    panel = Image.new("RGB", (base_image.width * 3, base_image.height + header), "white")
    panel.paste(base_image.convert("RGB"), (0, header))
    panel.paste(heatmap, (base_image.width, header))
    panel.paste(overlay, (base_image.width * 2, header))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), title, fill="black")
    draw.text((base_image.width + 8, 8), "heatmap", fill="black")
    draw.text((base_image.width * 2 + 8, 8), "overlay", fill="black")
    return panel


def _channel_grid(feature: torch.Tensor, top_channels: int) -> Tuple[Image.Image, List[int]]:
    clean = torch.nan_to_num(feature.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    if clean.ndim != 3:
        raise ValueError("Feature tensor must have shape [C, H, W]")
    count = min(max(top_channels, 1), clean.shape[0])
    scores = clean.flatten(1).std(dim=1, unbiased=False)
    indices = scores.topk(count).indices.tolist()

    tile = 144
    header = 22
    columns = min(4, count)
    rows = math.ceil(count / columns)
    canvas = Image.new("RGB", (columns * tile, rows * (tile + header)), "white")
    draw = ImageDraw.Draw(canvas)
    for position, channel_index in enumerate(indices):
        row, column = divmod(position, columns)
        values = clean[channel_index].numpy()
        normalized = _normalize_map(values)
        image = _colorize(normalized).resize((tile, tile), _resampling_bilinear())
        x, y = column * tile, row * (tile + header)
        canvas.paste(image, (x, y + header))
        draw.text(
            (x + 5, y + 4),
            f"ch {channel_index}  std={float(scores[channel_index]):.3g}",
            fill="black",
        )
    return canvas, indices


def select_debug_modules(
    model: nn.Module,
    scope: str = "key",
    include_patterns: Sequence[str] | None = None,
) -> Dict[str, nn.Module]:
    """Select named modules for key-stage or exhaustive leaf-layer tracing."""
    named = dict(model.named_modules())
    named.pop("", None)

    if include_patterns:
        patterns = [re.compile(pattern) for pattern in include_patterns]
        selected = {
            name: module
            for name, module in named.items()
            if any(pattern.search(name) for pattern in patterns)
        }
    elif scope == "all":
        selected = {
            name: module for name, module in named.items() if not list(module.children())
        }
    elif scope == "key":
        candidates = ["backbone.patch_embed"]
        stage_indices = getattr(getattr(model, "backbone", None), "stage_output_indices", ())
        candidates.extend(f"backbone.network.{index}" for index in stage_indices)
        norm_names = getattr(getattr(model, "backbone", None), "out_norm_names", ())
        candidates.extend(f"backbone.{name}" for name in norm_names)
        candidates.extend(
            [
                "fpn.l2_block",
                "fpn.l3_block",
                "fpn.l4_lateral",
                "fpn.l5_lateral",
                "fpn.p2_conv",
                "fpn.p3_conv",
                "fpn.p4_conv",
                "fpn.p5_conv",
                "fpn.p6_conv",
                "fpn",
                "head.cls_subnet",
                "head.cls_score",
                "head.reg_subnet",
                "head.reg_pred",
            ]
        )
        selected = {name: named[name] for name in candidates if name in named}
    else:
        raise ValueError("scope must be 'key' or 'all'")

    if not selected:
        raise ValueError("No module matched the requested debug layer selection")
    return selected


class FeatureMapRecorder:
    """Forward-hook recorder that writes compact diagnostics as layers execute."""

    def __init__(
        self,
        output_dir: str | Path,
        display_image: Image.Image,
        class_names: Sequence[str],
        num_anchors: int,
        *,
        top_channels: int = 16,
        save_visualizations: bool = True,
        save_tensors: bool = False,
    ):
        self.output_dir = Path(output_dir)
        max_display_side = 640
        display_scale = min(1.0, max_display_side / max(display_image.size))
        if display_scale < 1.0:
            display_image = display_image.resize(
                (
                    max(1, int(round(display_image.width * display_scale))),
                    max(1, int(round(display_image.height * display_scale))),
                ),
                _resampling_bilinear(),
            )
        self.display_image = display_image
        self.class_names = list(class_names)
        self.num_anchors = int(num_anchors)
        self.top_channels = int(top_channels)
        self.save_visualizations = bool(save_visualizations)
        self.save_tensors = bool(save_tensors)
        self.records: List[Dict[str, object]] = []
        self.class_records: List[Dict[str, object]] = []
        self.call_counts = defaultdict(int)
        self.handles = []
        self._record_index = 0

        for directory in ("aggregate", "channels", "class_maps", "tensors"):
            (self.output_dir / directory).mkdir(parents=True, exist_ok=True)

    def register(self, modules: Dict[str, nn.Module]) -> None:
        for name, module in modules.items():
            self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _make_hook(self, layer_name: str):
        def hook(module, _inputs, output):
            call_index = self.call_counts[layer_name]
            self.call_counts[layer_name] += 1
            for output_name, tensor in iter_output_tensors(output):
                self._record_tensor(layer_name, module, call_index, output_name, tensor)

        return hook

    def _record_tensor(
        self,
        layer_name: str,
        module: nn.Module,
        call_index: int,
        output_name: str,
        tensor: torch.Tensor,
    ) -> None:
        self._record_index += 1
        identifier = (
            f"{self._record_index:04d}_{_safe_name(layer_name)}"
            f"_call{call_index:02d}_{_safe_name(output_name)}"
        )
        stats = tensor_statistics(tensor)
        fpn_level = None
        if layer_name.startswith("head.") and call_index < len(FPN_LEVELS):
            fpn_level = FPN_LEVELS[call_index]
        else:
            level_match = re.search(r"(?:^|\.)p([2-6])(?:_|\.|$)", layer_name)
            if level_match:
                fpn_level = f"P{level_match.group(1)}"
            elif layer_name == "fpn" and output_name.startswith("output."):
                try:
                    output_index = int(output_name.rsplit(".", 1)[-1])
                except ValueError:
                    output_index = -1
                if 0 <= output_index < len(FPN_LEVELS):
                    fpn_level = FPN_LEVELS[output_index]
        record = {
            "index": self._record_index,
            "layer": layer_name,
            "module_type": type(module).__name__,
            "call": call_index,
            "fpn_level": fpn_level,
            "output": output_name,
            **stats,
            "aggregate_image": None,
            "channels_image": None,
            "tensor_file": None,
        }

        cpu = None
        if tensor.ndim == 4 and tensor.shape[0] > 0 and tensor.shape[1] > 0:
            cpu = tensor[0].detach().float().cpu()
            if self.save_visualizations:
                aggregate = cpu.abs().mean(dim=0).numpy()
                aggregate_path = self.output_dir / "aggregate" / f"{identifier}.png"
                title = f"{layer_name} call={call_index} shape={list(tensor.shape)} |mean(channel)|"
                _map_panel(aggregate, self.display_image, title).save(aggregate_path)
                record["aggregate_image"] = aggregate_path.relative_to(self.output_dir).as_posix()

                grid, indices = _channel_grid(cpu, self.top_channels)
                channels_path = self.output_dir / "channels" / f"{identifier}.png"
                grid.save(channels_path)
                record["channels_image"] = channels_path.relative_to(self.output_dir).as_posix()
                record["top_channels"] = indices

        if self.save_tensors:
            tensor_path = self.output_dir / "tensors" / f"{identifier}.pt"
            torch.save(tensor.detach().cpu(), tensor_path)
            record["tensor_file"] = tensor_path.relative_to(self.output_dir).as_posix()

        self.records.append(record)

        if layer_name == "head.cls_score" and tensor.ndim == 4:
            self._record_class_maps(tensor, call_index)

    def _record_class_maps(self, logits: torch.Tensor, call_index: int) -> None:
        batch, channels, height, width = logits.shape
        expected = self.num_anchors * len(self.class_names)
        if batch < 1 or channels != expected:
            self.class_records.append(
                {
                    "fpn_level": FPN_LEVELS[call_index] if call_index < len(FPN_LEVELS) else f"call_{call_index}",
                    "error": f"Expected {expected} channels, got {channels}",
                }
            )
            return

        level = FPN_LEVELS[call_index] if call_index < len(FPN_LEVELS) else f"call_{call_index}"
        probabilities = (
            torch.sigmoid(logits[0].detach().float().cpu())
            .reshape(self.num_anchors, len(self.class_names), height, width)
            .amax(dim=0)
        )
        for class_index, class_name in enumerate(self.class_names):
            values = probabilities[class_index]
            record = {
                "fpn_level": level,
                "class_id": class_index + 1,
                "class_name": class_name,
                "height": height,
                "width": width,
                "min_probability": float(values.min()),
                "mean_probability": float(values.mean()),
                "max_probability": float(values.max()),
                "image": None,
            }
            if self.save_visualizations:
                class_dir = self.output_dir / "class_maps" / level
                class_dir.mkdir(parents=True, exist_ok=True)
                path = class_dir / f"{class_index + 1:02d}_{_safe_name(class_name)}.png"
                title = (
                    f"{level} class={class_name} max_anchor_probability "
                    f"max={record['max_probability']:.4f} mean={record['mean_probability']:.4g}"
                )
                _map_panel(
                    values.numpy(),
                    self.display_image,
                    title,
                    fixed_probability_range=True,
                ).save(path)
                record["image"] = path.relative_to(self.output_dir).as_posix()
            self.class_records.append(record)

    def write_reports(self, metadata: Dict[str, object]) -> None:
        payload = {
            "metadata": metadata,
            "layers": self.records,
            "class_maps": self.class_records,
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True),
            encoding="utf-8",
        )

        fieldnames = [
            "index",
            "layer",
            "module_type",
            "call",
            "fpn_level",
            "output",
            "shape",
            "dtype",
            "numel",
            "min",
            "max",
            "mean",
            "std",
            "zero_fraction",
            "finite_fraction",
            "dead_channel_fraction",
            "flags",
            "aggregate_image",
            "channels_image",
            "tensor_file",
        ]
        with (self.output_dir / "layer_stats.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for record in self.records:
                row = dict(record)
                row["shape"] = json.dumps(row["shape"])
                row["flags"] = ",".join(row["flags"])
                writer.writerow(row)

        class_fields = [
            "fpn_level",
            "class_id",
            "class_name",
            "height",
            "width",
            "min_probability",
            "mean_probability",
            "max_probability",
            "image",
            "error",
        ]
        with (self.output_dir / "class_scores.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=class_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.class_records)

        self._write_html(metadata)

    def _write_html(self, metadata: Dict[str, object]) -> None:
        anomaly_count = sum(bool(record["flags"]) for record in self.records)
        rows = []
        for record in self.records:
            aggregate = (
                f'<a href="{html.escape(record["aggregate_image"])}">aggregate</a>'
                if record.get("aggregate_image")
                else "—"
            )
            channels = (
                f'<a href="{html.escape(record["channels_image"])}">channels</a>'
                if record.get("channels_image")
                else "—"
            )
            flags = ", ".join(record["flags"]) or "—"
            rows.append(
                "<tr>"
                f"<td>{record['index']}</td>"
                f"<td><code>{html.escape(record['layer'])}</code></td>"
                f"<td>{html.escape(record['module_type'])}</td>"
                f"<td>{record['call']}</td>"
                f"<td>{html.escape(str(record.get('fpn_level') or '—'))}</td>"
                f"<td><code>{html.escape(str(record['shape']))}</code></td>"
                f"<td>{record['mean']:.4g}</td><td>{record['std']:.4g}</td>"
                f"<td>{record['zero_fraction']:.3f}</td>"
                f"<td>{record['finite_fraction']:.3f}</td>"
                f"<td>{html.escape(flags)}</td><td>{aggregate} / {channels}</td>"
                "</tr>"
            )

        class_rows = []
        for record in self.class_records:
            if "error" in record:
                class_rows.append(
                    f"<tr><td>{html.escape(record['fpn_level'])}</td>"
                    f"<td colspan='5'>{html.escape(record['error'])}</td></tr>"
                )
                continue
            link = (
                f'<a href="{html.escape(record["image"])}">heatmap</a>'
                if record.get("image")
                else "—"
            )
            class_rows.append(
                "<tr>"
                f"<td>{html.escape(record['fpn_level'])}</td>"
                f"<td>{record['class_id']}</td>"
                f"<td>{html.escape(record['class_name'])}</td>"
                f"<td>{record['mean_probability']:.5f}</td>"
                f"<td>{record['max_probability']:.5f}</td><td>{link}</td>"
                "</tr>"
            )

        document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>FastViT feature debug</title>
<style>
body{{font-family:Arial,sans-serif;margin:24px;color:#222}} table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ddd;padding:6px;text-align:left;font-size:13px}} th{{background:#f3f3f3;position:sticky;top:0}}
code{{font-size:12px}} .warning{{color:#a33}} img{{max-width:100%}}
</style></head><body>
<h1>FastViT feature-map debug report</h1>
<p>Recorded <b>{len(self.records)}</b> tensors; <b class="warning">{anomaly_count}</b> have diagnostic flags.</p>
<p><a href="input_original.png">original input</a> · <a href="input_preprocessed.png">preprocessed input</a>
· <a href="predictions.png">predictions</a> · <a href="summary.json">JSON</a>
· <a href="layer_stats.csv">layer CSV</a> · <a href="class_scores.csv">class CSV</a></p>
<details><summary>Run metadata</summary><pre>{html.escape(json.dumps(metadata, indent=2, ensure_ascii=False))}</pre></details>
<h2>Layer activations</h2>
<table><thead><tr><th>#</th><th>Layer</th><th>Type</th><th>Call</th><th>FPN</th><th>Shape</th>
<th>Mean</th><th>Std</th><th>Zero</th><th>Finite</th><th>Flags</th><th>Views</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>Per-class classification maps</h2>
<p>Each map is sigmoid probability maximized over the anchors at one spatial location. Colours use a fixed 0–1 scale.</p>
<table><thead><tr><th>FPN</th><th>ID</th><th>Class</th><th>Mean probability</th><th>Max probability</th><th>View</th></tr></thead>
<tbody>{''.join(class_rows)}</tbody></table>
</body></html>"""
        (self.output_dir / "index.html").write_text(document, encoding="utf-8")


def serialize_predictions(
    prediction: Dict[str, torch.Tensor],
    image_metadata: Dict[str, object],
    class_names: Sequence[str],
) -> List[Dict[str, object]]:
    """Scale resized-space detections back to the original image."""
    scale_x = float(image_metadata["scale_x"])
    scale_y = float(image_metadata["scale_y"])
    boxes = prediction["boxes"].detach().cpu().float().clone()
    boxes[:, (0, 2)] /= scale_x
    boxes[:, (1, 3)] /= scale_y
    scores = prediction["scores"].detach().cpu()
    labels = prediction["labels"].detach().cpu()
    rows = []
    for box, score, label in zip(boxes, scores, labels):
        class_id = int(label)
        class_name = class_names[class_id - 1] if 1 <= class_id <= len(class_names) else "unknown"
        rows.append(
            {
                "box_xyxy": [float(value) for value in box],
                "score": float(score),
                "class_id": class_id,
                "class_name": class_name,
            }
        )
    return rows


def draw_predictions(
    image: Image.Image,
    predictions: Sequence[Dict[str, object]],
) -> Image.Image:
    result = image.copy().convert("RGB")
    draw = ImageDraw.Draw(result)
    palette = (
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
        "#a65628", "#f781bf", "#17becf", "#bcbd22", "#1f77b4",
    )
    for prediction in predictions:
        x1, y1, x2, y2 = prediction["box_xyxy"]
        colour = palette[(int(prediction["class_id"]) - 1) % len(palette)]
        draw.rectangle((x1, y1, x2, y2), outline=colour, width=3)
        label = f"{prediction['class_name']} {prediction['score']:.3f}"
        left, top, right, bottom = draw.textbbox((x1, y1), label)
        label_y = max(0, y1 - (bottom - top) - 4)
        draw.rectangle((x1, label_y, x1 + (right - left) + 6, y1), fill=colour)
        draw.text((x1 + 3, label_y + 1), label, fill="white")
    return result
