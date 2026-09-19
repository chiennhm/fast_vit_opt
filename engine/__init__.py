"""Training and evaluation utilities with lazy imports."""

from importlib import import_module

__all__ = [
    "WarmupCosineScheduler",
    "WarmupStepDecayScheduler",
    "evaluate",
    "load_checkpoint",
    "save_checkpoint",
    "train_one_epoch",
]

_EXPORTS = {
    "WarmupCosineScheduler": (".schedulers", "WarmupCosineScheduler"),
    "WarmupStepDecayScheduler": (".schedulers", "WarmupStepDecayScheduler"),
    "evaluate": (".evaluate", "evaluate"),
    "load_checkpoint": (".checkpoint", "load_checkpoint"),
    "save_checkpoint": (".checkpoint", "save_checkpoint"),
    "train_one_epoch": (".train", "train_one_epoch"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
