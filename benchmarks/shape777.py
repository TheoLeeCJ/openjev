"""Reproduce fresh versus parallel shared-state scoring on the owned 37x21 fixture."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import statistics
import time
from pathlib import Path

from semif_phase1.artifacts import write_new_outputs
from semif_phase1.core import load_causal_model
from semif_phase1.direct import score
from semif_phase1.serial import SerialPrefixScorer
from semif_phase1.shared import score_shared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    predictions_path = args.output.with_suffix(".predictions.jsonl")
    if any(
        path.exists() or path.is_symlink() for path in (args.output, predictions_path)
    ):
        parser.error("Report and predictions outputs must be new")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    if len(rows) != 777 or len(groups) != 37 or any(len(group) != 21 for group in groups.values()):
        parser.error("Expected the committed 37-state x 21-question fixture")
    model, tokenizer, metadata = load_causal_model(args.model, args.revision)
    import torch

    accelerator = getattr(torch, next(model.parameters()).device.type)

    first = next(iter(groups.values()))
    score(model, tokenizer, first[0], metadata, args.max_tokens)
    warm_serial = SerialPrefixScorer(model, tokenizer, metadata, args.max_tokens)
    for row in first:
        warm_serial.score(row)
    score_shared(model, tokenizer, first, metadata, args.max_tokens)
    report = {
        "version": "shape777-published-v1",
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "model": metadata,
        "hardware": accelerator.get_device_name(0),
        "timing_scope": "Warm model; includes prompt construction, tokenization, transfers, forward passes and CPU readout.",
        "results": [],
    }
    predictions = {}
    for mode in ("fresh", "serial_prefix", "parallel_shared"):
        accelerator.reset_peak_memory_stats()
        started = time.perf_counter()
        values, state_times = [], []
        for group in groups.values():
            mark = time.perf_counter()
            if mode == "fresh":
                values.extend(score(model, tokenizer, row, metadata, args.max_tokens) for row in group)
            elif mode == "serial_prefix":
                scorer = SerialPrefixScorer(model, tokenizer, metadata, args.max_tokens)
                values.extend(scorer.score(row) for row in group)
            else:
                scored, _ = score_shared(model, tokenizer, group, metadata, args.max_tokens)
                values.extend(scored)
            state_times.append(time.perf_counter() - mark)
        elapsed = time.perf_counter() - started
        predictions[mode] = values
        report["results"].append(
            {
                "mode": mode,
                "wall_seconds": elapsed,
                "decisions_per_second": len(values) / elapsed,
                "state_p50_seconds": statistics.median(state_times),
                "peak_cuda_bytes": accelerator.max_memory_allocated(),
            }
        )
    reference = {row["id"]: row for row in predictions["fresh"]}
    report["comparisons_to_fresh"] = {}
    for mode in ("serial_prefix", "parallel_shared"):
        flips, maximum = [], 0.0
        for row in predictions[mode]:
            old = reference[row["id"]]
            maximum = max(
                maximum, *(abs(a - b) for a, b in zip(old["probabilities"], row["probabilities"]))
            )
            if max(range(len(old["probabilities"])), key=old["probabilities"].__getitem__) != max(
                range(len(row["probabilities"])), key=row["probabilities"].__getitem__
            ):
                flips.append(row["id"])
        report["comparisons_to_fresh"][mode] = {
            "max_probability_difference": maximum,
            "argmax_flips": flips,
        }
    write_new_outputs({
        args.output: json.dumps(report, indent=2, allow_nan=False) + "\n",
        predictions_path: "".join(
            json.dumps({"mode": mode, **row}, allow_nan=False) + "\n"
            for mode, values in predictions.items()
            for row in values
        ),
    })
    print(json.dumps(report["results"]))


if __name__ == "__main__":
    main()
