# Intel Arc (XPU)

The Torch backend runs SemIf's direct, serial-prefix reuse, and parallel
shared-state decision modes on an Intel Arc discrete GPU. CUDA stays the
default backend. The loader picks CUDA when CUDA is available. The loader
tries an Intel GPU only when no CUDA device is available. This applies to
any Arc card with enough VRAM to hold the model. The validation below used
one Intel Arc A770 with 16 GB of VRAM.

## Install

Install the XPU build of PyTorch first, from its own package index. Then
install the project:

```bash
pip install torch==2.10.0+xpu --index-url https://download.pytorch.org/whl/xpu
pip install -e '.[test]'
```

The validated versions are `torch==2.10.0+xpu` and `transformers==5.17.0`,
on Python 3.10. Use the same model revisions as the CUDA path. The `xpu`
extra in `pyproject.toml` records this torch pin. It does not fetch from
the separate wheel index; run the two commands above in this order.

## Select one device

Set `ONEAPI_DEVICE_SELECTOR` to expose exactly one Intel GPU per scorer
process:

```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:1
```

Use the index for your own GPU. The index can differ between machines and
between driver updates. Do not assume the index from another guide or from
a past run. Check the device name first:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:1 python -c \
  "import torch; [print(i, torch.xpu.get_device_properties(i).name) for i in range(torch.xpu.device_count())]"
```

The loader raises an error when zero or more than one CUDA or XPU device is
visible. The error message names both `CUDA_VISIBLE_DEVICES` and
`ONEAPI_DEVICE_SELECTOR`.

## The long-forward workaround

On XPU, one long forward pass can corrupt the last-token logits. The
failure starts near 1813 input tokens on the Arc A770 test system, on the
Qwen3.5 hybrid architecture. The cause is upstream in PyTorch XPU, not in
this project.

The direct, serial, and shared scorers all split a long XPU forward into
short steps of 1024 tokens. Each step reuses the model's own KV cache. The
final logits come from the last step. This chunked path gives the same
result as one full forward on CPU. The workaround runs on XPU only. It does
not change the CUDA or CPU path in any way.

## Generation settings for XPU

The compact-generation benchmark needs two extra settings on XPU:

```bash
python benchmarks/decision_vs_generation.py \
  --prefill-chunk-size 512 --attention eager \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input benchmarks/data/shape777.jsonl \
  --output /path/to/new-generation-a770.json
```

`--prefill-chunk-size 512` keeps every internal prefill forward under the
corruption length. `--attention eager` avoids a separate decode-side defect
that repeats one token under `sdpa` on XPU. With both settings, generation
reproduced the published 21-item array exactly. Both flags default to
upstream behavior when omitted; only XPU runs need them.

## Reranker drift on XPU

The reranker benchmark runs on XPU and completes. Its choices drift from
the published NVIDIA choices. The reranker readout compares two large
yes/no logits, near 17 in magnitude. Their difference is small, often
between 0.1 and 1.0. Small BF16 rounding differences can flip that small
difference's sign. A CPU FP32 reference matched the published NVIDIA
choices, not the A770 choices. This points to readout sensitivity, not a
port bug. Treat reranker output on XPU as informational only. Do not use it
as a matched reproduction of the published reranker result.

## What this port validates

Validated: direct, serial, and shared scoring reach drift grade against the
published NVIDIA rows. Between 3 and 7 of 777 choices differ, with a
maximum probability difference near 0.09. This sits inside the repository's
own same-GPU drift range. Compact generation with the two settings above
reproduced the published 21-item array exactly.

Limited: reranker choices drift on XPU, for the readout-sensitivity reason
above. Treat reranker output on XPU as informational only.

Untested: other Arc cards, multiple GPUs, Windows, and torch versions other
than `2.10.0+xpu`. The Arc B580 is expected to work but was not the
validation target.

## Checks

Run these from the repository root, in an environment installed with
`pip install -e '.[test]'`. No GPU is needed:

```bash
pytest -q
(cd results/raw && sha256sum -c SHA256SUMS)
python benchmarks/verify_published.py
```

Run this check against the committed A770 evidence bundle:

```bash
(cd results/xpu && sha256sum -c SHA256SUMS)
```

See [the A770 evidence](../results/xpu/README.md) for the machine, the
dates, and the claim boundaries behind the numbers above.
