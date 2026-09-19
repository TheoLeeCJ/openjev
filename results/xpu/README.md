# Intel Arc A770 evidence

These runs use one Intel Arc A770 with 16 GB of VRAM, `torch==2.10.0+xpu`,
and `transformers==5.17.0`, selected with `ONEAPI_DEVICE_SELECTOR`. The
A770 was the boot GPU on a shared desktop machine; no other GPU load ran
during a timed pass. The runs are dated 2026-09-18. Timings describe this
one machine and this one workload. They are not a cross-hardware claim
against the published RTX 3090 numbers; only the choice comparisons below
make that claim, and only at drift grade.

The method and the review threshold are frozen in [manifest.json](manifest.json)
before any comparison below. Every changed choice is reported.

## Direct, serial, and shared scoring

The full 37-state by 21-question fixture (777 decisions) ran in `fresh`,
`serial_prefix`, and `parallel_shared` modes.

| Mode | Argmax flips vs published NVIDIA rows | Maximum probability difference |
|---|---:|---:|
| `fresh` | 7 / 777 | 0.0865 |
| `serial_prefix` | 4 / 777 | 0.0622 |
| `parallel_shared` | 3 / 777 | 0.1131 |

The published NVIDIA run itself has 5 / 777 flips between its own `fresh`
and `serial` modes. The A770 result sits inside that same drift range for
this BF16 workload.

Evidence: [shape777.json](2026-09-18-a770/shape777.json),
[row-level predictions](2026-09-18-a770/shape777.jsonl.gz).

## Compact generation

A 21-decision compact-generation run used `--prefill-chunk-size 512
--attention eager`, the settings [docs/XPU.md](../../docs/XPU.md)
documents for XPU. All three repeats returned the same valid 21-item JSON
array, and its choices matched the published NVIDIA array exactly.
Agreement with direct argmax was 18 / 21, the published value.

Median direct scoring took 8.88 s; median generation took 14.43 s (1.62x
as long). This is a systems comparison on one machine, not a claim that
the two readouts are semantically equivalent.

Evidence: [generation.json](2026-09-18-a770/generation.json).

## Reranker (informational only)

The reranker benchmark ran on the same fixture at pair batch sizes 1, 4,
and 8. It completed without error but did not reproduce the published
NVIDIA row-level choices closely: 138 / 777 choices differ at batch size
1, 329 / 777 at batch size 4, and 341 / 777 at batch size 8.

The reranker readout compares two large yes/no logits, near 17 in
magnitude, with a small difference, often between 0.1 and 1.0. A CPU FP32
reference confirmed the published NVIDIA choices and did not confirm the
A770 choices, which points to readout sensitivity rather than a port
defect. Do not use this reranker result as a matched reproduction. Treat
it as informational only.

Evidence: [reranker.json](2026-09-18-a770/reranker.json),
[row-level predictions](2026-09-18-a770/reranker.jsonl.gz).

## CLI smoke outputs

Four installed-CLI direct runs on real pinned checkpoints (Qwen3-0.6B,
MiniCPM5-2B, Qwen3.5-4B, and one reranker-mode smoke run) each returned
finite normalized option scores on the same three fixture rows.

Evidence: [smoke.jsonl](2026-09-18-a770/smoke.jsonl).

## Storage

Row-level prediction files are stored as losslessly compressed
`.jsonl.gz` files, made with `gzip -n` for deterministic bytes. Reports
and the smoke file stay plain text. To inspect a compressed file:

```bash
gzip -dc results/xpu/2026-09-18-a770/shape777.jsonl.gz
```

## Validation

```bash
(cd results/xpu && sha256sum -c SHA256SUMS)
```

See [Intel Arc (XPU)](../../docs/XPU.md) for the port, the install, and
the known limits behind these numbers.
