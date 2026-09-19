"""Parallel decisions over one exact shared state using a native prefix cache."""

from __future__ import annotations

import inspect
import json
import time

from .core import direct_messages, softmax, synchronize_device
from .direct import PROMPT_VERSION, cached_forward, encode_prompt


def _state_prefix(tokenizer, state) -> list[int]:
    row = {
        "id": "prefix-only",
        "state": state,
        # This value occurs after the extracted evidence boundary.
        "question": "prefix boundary placeholder",
        "options": [
            {"id": "yes", "description": "Yes"},
            {"id": "no", "description": "No"},
        ],
    }
    turns = direct_messages(row)
    prompt = tokenizer.apply_chat_template(
        turns, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    payload = turns[-1]["content"]
    if prompt.count(payload) != 1:
        raise ValueError("Cannot locate the unmodified evidence payload in the chat template")
    evidence = json.dumps({"evidence": state}, ensure_ascii=False)[:-1]
    if not payload.startswith(evidence):
        raise ValueError("Evidence serialization changed")
    text = prompt[: prompt.index(payload)] + evidence
    # Appending JSON punctuation can merge with the final boundary token.
    return tokenizer.encode(text, add_special_tokens=False)[:-1]


def _suffix_layout(sequences: list[list[int]], prefix_length: int, pad_id: int):
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("Every decision needs a nonempty suffix")
    width = max(map(len, sequences))
    ids, masks, positions, ends = [], [], [], []
    for sequence in sequences:
        padding = width - len(sequence)
        ids.append(sequence + [pad_id] * padding)
        masks.append([1] * (prefix_length + len(sequence)) + [0] * padding)
        positions.append(list(range(prefix_length, prefix_length + len(sequence))) + [0] * padding)
        ends.append(len(sequence) - 1)
    return {"input_ids": ids, "attention_mask": masks, "position_ids": positions}, ends


def score_shared(model, tokenizer, rows: list[dict], metadata: dict, max_tokens: int = 4096):
    """Return all option distributions together after one state prefill."""
    import torch

    if not rows or any(row["state"] != rows[0]["state"] for row in rows[1:]):
        raise ValueError("Shared scoring requires one nonempty exact state")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Decision IDs must be unique")
    started = time.perf_counter()
    encoded = [encode_prompt(tokenizer, row, max_tokens) for row in rows]
    prefix = _state_prefix(tokenizer, rows[0]["state"])
    if not prefix or any(ids[: len(prefix)] != prefix or len(ids) <= len(prefix) for ids, _, _ in encoded):
        raise ValueError("The fixed state prefix does not match every full prompt")
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        raise ValueError("Tokenizer requires a padding or EOS token")
    layout, ends = _suffix_layout([ids[len(prefix) :] for ids, _, _ in encoded], len(prefix), pad)
    selected_positions = sorted(set(ends))
    encode_seconds = time.perf_counter() - started
    device = next(model.parameters()).device
    sync = lambda: synchronize_device(device)
    parameters = inspect.signature(model.forward).parameters
    if "logits_to_keep" not in parameters and hasattr(model, "get_base_model"):
        parameters = inspect.signature(model.get_base_model().forward).parameters
    if "logits_to_keep" not in parameters:
        raise RuntimeError("Model lacks selective-position logits needed by shared scoring")
    model.eval()
    with torch.inference_mode():
        sync()
        mark = time.perf_counter()
        output = cached_forward(
            model,
            {
                "input_ids": torch.tensor([prefix], dtype=torch.long, device=device),
                "attention_mask": torch.ones((1, len(prefix)), dtype=torch.long, device=device),
            },
        )
        cache = output.past_key_values
        del output
        sync()
        prefill_seconds = time.perf_counter() - mark
        if cache is None or cache.get_seq_length() != len(prefix):
            raise RuntimeError("Invalid native prefix cache")
        if not callable(getattr(cache, "reorder_cache", None)):
            raise RuntimeError("Native cache does not support duplicate branch selection")
        sync()
        mark = time.perf_counter()
        cache.reorder_cache(torch.zeros(len(rows), dtype=torch.long, device=device))
        sync()
        replicate_seconds = time.perf_counter() - mark
        inputs = {key: torch.tensor(value, dtype=torch.long, device=device) for key, value in layout.items()}
        sync()
        mark = time.perf_counter()
        output = model(
            **inputs,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=torch.tensor(selected_positions, dtype=torch.long, device=device),
        )
        sync()
        suffix_seconds = time.perf_counter() - mark
        results = []
        for index, (row, (ids, slots, prompt_hash)) in enumerate(zip(rows, encoded)):
            vocabulary = output.logits[index, selected_positions.index(ends[index]), :].float()
            selected = vocabulary[slots].cpu().tolist()
            results.append(
                {
                    "id": row["id"],
                    "option_ids": [option["id"] for option in row["options"]],
                    "probabilities": softmax(selected),
                    "option_logits": selected,
                    "input_tokens": len(ids),
                    "prompt_sha256": prompt_hash,
                    "prompt_version": PROMPT_VERSION,
                    "model": {**metadata, "serving_config": "native-state-prefix-parallel-v1"},
                    "readout": "native selected suffix-position logits",
                    "probability_status": "conditional option score; uncalibrated as decision confidence",
                }
            )
        del output, cache
    sync()
    timing = {
        "total_seconds": time.perf_counter() - started,
        "encode_seconds": encode_seconds,
        "prefix_tokens": len(prefix),
        "prefill_seconds": prefill_seconds,
        "replicate_seconds": replicate_seconds,
        "suffix_forward_seconds": suffix_seconds,
        "batch_size": len(rows),
        "true_suffix_tokens": sum(len(ids) - len(prefix) for ids, _, _ in encoded),
        "padded_suffix_tokens": len(rows) * len(layout["input_ids"][0]),
    }
    return results, timing
