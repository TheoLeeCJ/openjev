"""Direct categorical decision readout from native next-token logits."""

from __future__ import annotations

import inspect
import time

from .core import LETTERS, digest, direct_messages, softmax, synchronize_device

PROMPT_VERSION = "direct-options-v1"


def _slot_ids(tokenizer, count: int) -> list[int]:
    result = []
    for letter in LETTERS[:count]:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != letter:
            raise ValueError(f"Answer slot {letter!r} is not one exact round-trip token")
        result.append(encoded[0])
    if len(result) != len(set(result)):
        raise ValueError("Answer-slot tokens collide")
    return result


# On XPU, one long forward can corrupt the last-token logits.
# This failure occurs near 1813 tokens on Arc A770 with Qwen3.5-4B.
# Split long XPU inputs into short KV-cache steps.
# The CPU result is bit-identical to one full forward.
_XPU_FORWARD_TOKEN_LIMIT = 1024


def _forward(model, inputs):
    parameters = inspect.signature(model.forward).parameters
    kwargs = dict(inputs, use_cache=False, return_dict=True)
    if "logits_to_keep" in parameters:
        kwargs["logits_to_keep"] = 1
    ids = inputs["input_ids"]
    if ids.device.type != "xpu" or ids.shape[1] <= _XPU_FORWARD_TOKEN_LIMIT:
        return model(**kwargs).logits[:, -1, :]
    cache = None
    for start in range(0, ids.shape[1], _XPU_FORWARD_TOKEN_LIMIT):
        piece = ids[:, start : start + _XPU_FORWARD_TOKEN_LIMIT]
        call = {
            "input_ids": piece,
            "attention_mask": inputs["attention_mask"][:, : start + piece.shape[1]],
            "use_cache": True,
            "return_dict": True,
            "past_key_values": cache,
        }
        if "logits_to_keep" in parameters:
            call["logits_to_keep"] = 1
        output = model(**call)
        cache = output.past_key_values
        if cache is None:
            raise RuntimeError("Long XPU scoring requires a native KV cache")
    return output.logits[:, -1, :]


def cached_forward(model, inputs):
    """Return one cached model output, chunking long XPU inputs through the KV cache."""
    parameters = inspect.signature(model.forward).parameters
    if "logits_to_keep" not in parameters and hasattr(model, "get_base_model"):
        parameters = inspect.signature(model.get_base_model().forward).parameters
    accepts = "logits_to_keep" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    if not accepts:
        raise RuntimeError("Model lacks selective-position logits needed by cached scoring")
    ids = inputs["input_ids"]
    if ids.device.type != "xpu" or ids.shape[1] <= _XPU_FORWARD_TOKEN_LIMIT:
        return model(**inputs, use_cache=True, return_dict=True, logits_to_keep=1)
    cache = inputs.get("past_key_values")
    base = {key: value for key, value in inputs.items() if key != "past_key_values"}
    # The mask covers the existing cache plus the new input, so chunk cuts must
    # keep the cache region and grow only by the tokens processed so far.
    offset = inputs["attention_mask"].shape[1] - ids.shape[1]
    if offset < 0:
        raise ValueError("Attention mask cannot be shorter than the input")
    output = None
    for start in range(0, ids.shape[1], _XPU_FORWARD_TOKEN_LIMIT):
        piece = ids[:, start : start + _XPU_FORWARD_TOKEN_LIMIT]
        call = {
            **base,
            "input_ids": piece,
            "attention_mask": inputs["attention_mask"][:, : offset + start + piece.shape[1]],
            "use_cache": True,
            "return_dict": True,
            "past_key_values": cache,
            "logits_to_keep": 1,
        }
        output = model(**call)
        cache = output.past_key_values
        if cache is None:
            raise RuntimeError("Long XPU scoring requires a native KV cache")
    return output


def encode_prompt(tokenizer, row: dict, max_tokens: int) -> tuple[list[int], list[int], str]:
    """Encode one decision and verify its single-token answer slots."""
    prompt = tokenizer.apply_chat_template(
        direct_messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids or len(ids) > max_tokens:
        raise ValueError(f"Row {row['id']}: {len(ids)} input tokens exceed limit {max_tokens}; no truncation allowed")
    slots = _slot_ids(tokenizer, len(row["options"]))
    for letter, token in zip(LETTERS, slots):
        if tokenizer.encode(prompt + letter, add_special_tokens=False) != ids + [token]:
            raise ValueError(f"Answer boundary changes tokenization for slot {letter}")
    return ids, slots, digest(prompt)


def score(model, tokenizer, row: dict, metadata: dict, max_tokens: int = 4096) -> dict:
    import torch

    started = time.perf_counter()
    ids, slots, prompt_hash = encode_prompt(tokenizer, row, max_tokens)
    device = next(model.parameters()).device
    inputs = {
        "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=device),
    }
    synchronize_device(device)
    forward_start = time.perf_counter()
    with torch.inference_mode():
        vocabulary = _forward(model, inputs)[0].float()
    synchronize_device(device)
    selected = vocabulary[slots].cpu().tolist()
    return {
        "id": row["id"],
        "option_ids": [option["id"] for option in row["options"]],
        "probabilities": softmax(selected),
        "option_logits": selected,
        "input_tokens": len(ids),
        "forward_seconds": time.perf_counter() - forward_start,
        "total_seconds": time.perf_counter() - started,
        "prompt_sha256": prompt_hash,
        "prompt_version": PROMPT_VERSION,
        "model": metadata,
        "readout": "native full-vocabulary last-position logits restricted to declared answer slots",
        "probability_status": "conditional option score; uncalibrated as decision confidence",
    }
