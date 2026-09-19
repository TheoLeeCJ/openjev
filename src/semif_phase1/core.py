"""Shared input validation, prompts, model loading, and numeric helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOP"
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)


def validate_row(row: dict) -> None:
    required = {"id", "state", "question", "options"}
    if not required <= row.keys():
        raise ValueError(f"Row is missing fields: {sorted(required - row.keys())}")
    if not all(isinstance(row[key], str) and row[key] for key in ("id", "question")):
        raise ValueError("id and question must be nonempty strings")
    state = row["state"]
    if not isinstance(state, (str, dict, list)) or not state:
        raise ValueError("state must be a nonempty string, object, or array")
    try:
        json.dumps(state, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("state must be finite JSON-compatible data") from error
    options = row["options"]
    if not isinstance(options, list) or not 2 <= len(options) <= len(LETTERS):
        raise ValueError("options must contain 2-16 entries")
    ids = []
    for option in options:
        if not isinstance(option, dict) or not isinstance(option.get("id"), str) or not isinstance(option.get("description"), str):
            raise ValueError("Each option needs string id and description fields")
        ids.append(option["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Option IDs must be unique")


def direct_messages(row: dict) -> list[dict]:
    validate_row(row)
    payload = {
        "evidence": row["state"],
        "criterion": row["question"],
        "options": [
            {"letter": LETTERS[index], "description": option["description"]}
            for index, option in enumerate(row["options"])
        ],
    }
    return [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def softmax(values: list[float]) -> list[float]:
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        raise ValueError("Need at least two finite scores")
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def synchronize_device(device) -> None:
    """Synchronize a CUDA or XPU device. Other devices need no synchronization."""
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device)


def load_causal_model(source: str, revision: str, attn_implementation: str = "sdpa", backend: str | None = None):
    """Load one pinned causal model on the sole visible accelerator device.

    Pass backend="cuda" or backend="xpu" to require that accelerator. Leave
    it None to auto-detect: CUDA first, then XPU.
    """
    import torch
    import transformers

    local = Path(source).exists()
    if not local and not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("Remote models require a pinned 40-character commit revision")
    if local and not revision:
        raise ValueError("Local models require an explicit manifest/revision string")
    if backend not in (None, "cuda", "xpu"):
        raise ValueError("backend must be 'cuda', 'xpu', or None for auto-detection")
    has_cuda = torch.cuda.is_available()
    has_xpu = getattr(torch, "xpu", None) is not None and torch.xpu.is_available()
    if backend == "cuda":
        if not has_cuda:
            raise ValueError("This scorer requires one CUDA GPU")
    elif backend == "xpu":
        if not has_xpu:
            raise ValueError("This scorer requires one XPU GPU")
    elif has_cuda:
        backend = "cuda"
    elif has_xpu:
        backend = "xpu"
    else:
        raise ValueError("This scorer requires one CUDA or XPU GPU")
    count = torch.cuda.device_count() if backend == "cuda" else torch.xpu.device_count()
    if count != 1:
        raise ValueError(
            "Expose exactly one GPU, for example with CUDA_VISIBLE_DEVICES or ONEAPI_DEVICE_SELECTOR"
        )
    common = {"revision": None if local else revision, "local_files_only": local, "trust_remote_code": False}
    config = transformers.AutoConfig.from_pretrained(source, **common)
    tokenizer = transformers.AutoTokenizer.from_pretrained(source, **common)
    cls = transformers.AutoModelForCausalLM
    if config.model_type in {"qwen3_5", "qwen3_5_text"}:
        cls = getattr(transformers, "Qwen3_5ForCausalLM", None)
        if cls is None:
            raise RuntimeError("Installed transformers lacks the native Qwen3.5 model")
        config = config.get_text_config()
    model, loading = cls.from_pretrained(
        source,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": f"{backend}:0"},
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
        output_loading_info=True,
        **common,
    )
    if any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Checkpoint did not load completely: {loading}")
    model.eval()
    metadata = {
        "source": source,
        "revision": revision,
        "dtype": "bfloat16",
        "attention": attn_implementation,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    return model, tokenizer, metadata
