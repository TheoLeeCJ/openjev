"""Serialized decisions over one-state native prefix caches."""

from __future__ import annotations

import copy
import hashlib
import json
import time

from .core import direct_messages, softmax, synchronize_device
from .direct import PROMPT_VERSION, cached_forward, encode_prompt


def _state_prefix(tokenizer, state) -> list[int]:
    row = {
        "id": "prefix-only",
        "state": state,
        "question": "prefix boundary placeholder",
        "options": [{"id": "yes", "description": "Yes"}, {"id": "no", "description": "No"}],
    }
    turns = direct_messages(row)
    prompt = tokenizer.apply_chat_template(
        turns, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    payload = turns[-1]["content"]
    evidence = json.dumps({"evidence": state}, ensure_ascii=False)[:-1]
    if prompt.count(payload) != 1 or not payload.startswith(evidence):
        raise ValueError("Cannot establish a deterministic evidence prefix")
    text = prompt[: prompt.index(payload)] + evidence
    return tokenizer.encode(text, add_special_tokens=False)[:-1]


def _cached_forward(model, inputs):
    return cached_forward(model, inputs)


class SerialPrefixScorer:
    """Cache the current state, then score independent copied suffix branches."""

    def __init__(self, model, tokenizer, metadata: dict, max_tokens: int = 4096):
        self.model = model
        self.tokenizer = tokenizer
        self.metadata = {**metadata, "serving_config": "native-state-prefix-cache-v1"}
        self.max_tokens = max_tokens
        self.device = next(model.parameters()).device
        self.cache = None
        self.state = None
        self.prefix = None

    def score(self, row: dict) -> dict:
        import torch

        started = time.perf_counter()
        ids, slots, prompt_hash = encode_prompt(self.tokenizer, row, self.max_tokens)
        hit = self.cache is not None and row["state"] == self.state
        prefix = self.prefix if hit else _state_prefix(self.tokenizer, row["state"])
        if not prefix or ids[: len(prefix)] != prefix or len(ids) <= len(prefix):
            raise ValueError("State prefix does not match the full prompt")
        sync = lambda: synchronize_device(self.device)
        prefill_seconds = 0.0
        self.model.eval()
        with torch.inference_mode():
            if not hit:
                self.cache = self.state = self.prefix = None
                sync()
                mark = time.perf_counter()
                output = _cached_forward(
                    self.model,
                    {
                        "input_ids": torch.tensor([prefix], dtype=torch.long, device=self.device),
                        "attention_mask": torch.ones((1, len(prefix)), dtype=torch.long, device=self.device),
                    },
                )
                self.cache = output.past_key_values
                del output
                sync()
                prefill_seconds = time.perf_counter() - mark
                if self.cache is None or self.cache.get_seq_length() != len(prefix):
                    raise RuntimeError("Invalid native prefix cache")
                self.state, self.prefix = row["state"], prefix
            sync()
            mark = time.perf_counter()
            branch = copy.deepcopy(self.cache)
            sync()
            copy_seconds = time.perf_counter() - mark
            inputs = {
                "input_ids": torch.tensor([ids[len(prefix) :]], dtype=torch.long, device=self.device),
                "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=self.device),
                "past_key_values": branch,
            }
            sync()
            mark = time.perf_counter()
            output = _cached_forward(self.model, inputs)
            sync()
            suffix_seconds = time.perf_counter() - mark
            vocabulary = output.logits[0, -1, :].float()
            selected_tensor = vocabulary[slots]
            selected = selected_tensor.cpu().tolist()
            result = {
                "id": row["id"],
                "option_ids": [option["id"] for option in row["options"]],
                "probabilities": softmax(selected),
                "option_logits": selected,
                "answer_token_ids": slots,
                "input_tokens": len(ids),
                "prompt_sha256": prompt_hash,
                "prompt_version": PROMPT_VERSION,
                "model": self.metadata,
                "readout": "native-state-prefix-cache-last-position",
                "probability_status": "conditional option score; uncalibrated as decision confidence",
                "cache_hit": hit,
                "prefix_tokens": len(prefix),
                "prefix_sha256": hashlib.sha256(json.dumps(prefix).encode()).hexdigest(),
                "allowed_token_mass": float(
                    (selected_tensor.logsumexp(-1) - vocabulary.logsumexp(-1)).exp()
                ),
                "full_vocab_argmax_id": int(vocabulary.argmax()),
                "prefill_seconds": prefill_seconds,
                "copy_seconds": copy_seconds,
                "suffix_forward_seconds": suffix_seconds,
            }
            del output, branch
        sync()
        result["forward_seconds"] = prefill_seconds + suffix_seconds
        result["total_seconds"] = time.perf_counter() - started
        return result
