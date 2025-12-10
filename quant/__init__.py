"""Utility entry points for quantization helpers."""
from __future__ import annotations

from typing import Any

from .quant_envs import AttrDict, Base_GRUQuantEnv
from .modules.ops import Sqrt, Pow, Add, Mul

__all__ = ["get_quant_model", "AttrDict", "Base_GRUQuantEnv", "Sqrt", "Pow", "Add", "Mul"]


def _build_quant_args(proj: Any) -> AttrDict:
    # Try to get from proj.args first, then fall back to proj attributes
    args = getattr(proj, "args", None)
    if args is not None:
        n_bits_w = getattr(args, "n_bits_w", 8)
        n_bits_a = getattr(args, "n_bits_a", 8)
        pretrained_model = getattr(args, "pretrained_model", "")
        quant_dir_label = getattr(args, "quant_dir_label", "")
    else:
        n_bits_w = getattr(proj, "n_bits_w", 8)
        n_bits_a = getattr(proj, "n_bits_a", 8)
        pretrained_model = getattr(proj, "pretrained_model", "")
        quant_dir_label = getattr(proj, "quant_dir_label", "")
    
    return AttrDict({
        "n_bits_w": n_bits_w,
        "n_bits_a": n_bits_a,
        "pretrained_model": pretrained_model,
        "quant_dir_label": quant_dir_label,
    })


def get_quant_model(proj: Any, model) -> Any:
    """Return a (possibly) quantized version of ``model``.

    If ``proj.quant`` is truthy we attempt to construct the quantization
    environment defined in ``quant.quant_envs``. On failure we log a warning and
    return ``model`` unchanged so callers do not crash in non-quant workflows.
    """
    if not getattr(proj, "quant", False):
        return model

    try:
        quant_args = _build_quant_args(proj)
        env = Base_GRUQuantEnv(model, args=quant_args)
        setattr(proj, "quant_env", env)
        return env.q_model
    except Exception as exc:  # pragma: no cover - protective fallback
        print(f"[WARN] Quantization setup failed: {exc}. Using float model instead.")
        return model
