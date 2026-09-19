"""BDD100K detection components with lazy imports."""

from importlib import import_module

__all__ = ["FastViTDetector", "DetectionLoss", "evaluate_coco", "draw_detections"]

_EXPORTS = {
    "FastViTDetector": (".fastvit_detector", "FastViTDetector"),
    "DetectionLoss": (".losses", "DetectionLoss"),
    "evaluate_coco": (".metrics", "evaluate_coco"),
    "draw_detections": (".visualize", "draw_detections"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
