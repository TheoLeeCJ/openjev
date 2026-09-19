"""Exercise accelerator contracts without requiring a GPU or model download."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from semif_phase1.core import load_causal_model, synchronize_device
from semif_phase1.direct import _forward, cached_forward


@pytest.mark.parametrize("backend,count", [("cuda", 1), ("xpu", 1), ("xpu", 2)])
def test_loader_backend_and_single_device_contract(monkeypatch, backend, count):
    accelerators = {
        name: SimpleNamespace(is_available=lambda name=name: name == backend,
                              device_count=lambda: count)
        for name in ("cuda", "xpu")
    }
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        **accelerators, bfloat16="bf16", __version__="test"))
    model = Mock()
    loader = Mock(return_value=(model, {}))
    transformers = SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=Mock(return_value=SimpleNamespace(model_type="qwen3"))),
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock()),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=loader),
        __version__="test",
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    if count != 1:
        with pytest.raises(ValueError, match="exactly one GPU"):
            load_causal_model("test/remote", "a" * 40)
        loader.assert_not_called()
    else:
        load_causal_model("test/remote", "a" * 40)
        assert loader.call_args.kwargs["device_map"] == {"": f"{backend}:0"}
        assert loader.call_args.kwargs["attn_implementation"] == "sdpa"
        assert loader.call_args.kwargs["revision"] == "a" * 40


def test_synchronization_dispatch(monkeypatch):
    cuda, xpu = Mock(), Mock()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda, xpu=xpu))
    for name in ("cpu", "cuda", "xpu"):
        synchronize_device(SimpleNamespace(type=name))
    assert cuda.synchronize.call_count == xpu.synchronize.call_count == 1


class XpuInput:
    """Select the XPU branch while performing model arithmetic on the CPU."""

    device = SimpleNamespace(type="xpu")

    def __init__(self, tensor):
        self.tensor = tensor
        self.shape = tensor.shape

    def __getitem__(self, key):
        return self.tensor[key]


@pytest.mark.parametrize("length", [1025, 2049])
def test_chunked_forward_matches_native_full_context(length):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
    )).eval()
    ids = torch.randint(0, 32, (1, length))
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        expected = _forward(model, {"input_ids": ids, "attention_mask": mask})
        actual = _forward(model, {"input_ids": XpuInput(ids), "attention_mask": mask})
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_chunked_forward_rejects_missing_cache():
    import torch

    class NoCache:
        def forward(self, **kwargs):
            return SimpleNamespace(past_key_values=None)

        __call__ = forward

    ids = torch.ones((1, 1025), dtype=torch.long)
    with pytest.raises(RuntimeError, match="native KV cache"):
        _forward(NoCache(), {"input_ids": XpuInput(ids), "attention_mask": ids})


def _tiny_qwen3():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    return Qwen3ForCausalLM(Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
    )).eval()


@pytest.mark.parametrize("length", [1025, 2049])
def test_chunked_cached_forward_matches_single_cache(length):
    import torch

    model = _tiny_qwen3()
    ids = torch.randint(0, 32, (1, length))
    mask = torch.ones_like(ids)
    suffix = torch.randint(0, 32, (1, 7))
    with torch.inference_mode():
        reference = model(
            input_ids=torch.cat([ids, suffix], dim=1), attention_mask=torch.ones((1, length + 7)),
            use_cache=True, return_dict=True, logits_to_keep=1,
        ).logits[:, -1, :]
        cache = cached_forward(model, {"input_ids": ids, "attention_mask": mask}).past_key_values
        assert cache.get_seq_length() == length
        actual = cached_forward(model, {
            "input_ids": suffix, "attention_mask": torch.ones((1, length + 7)),
            "past_key_values": cache,
        }).logits[:, -1, :]
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)


def test_chunked_cached_forward_rejects_missing_cache():
    import torch

    class NoCache:
        def forward(self, **kwargs):
            return SimpleNamespace(past_key_values=None)

        __call__ = forward

    ids = torch.ones((1, 1025), dtype=torch.long)
    with pytest.raises(RuntimeError, match="native KV cache"):
        cached_forward(NoCache(), {"input_ids": XpuInput(ids), "attention_mask": ids})


def test_chunked_cached_suffix_threads_existing_cache():
    import torch

    model = _tiny_qwen3()
    prefix = torch.randint(0, 32, (1, 700))
    suffix = torch.randint(0, 32, (1, 1500))
    ids = torch.cat([prefix, suffix], dim=1)
    with torch.inference_mode():
        reference = model(
            input_ids=ids, attention_mask=torch.ones((1, 2200)),
            use_cache=True, return_dict=True, logits_to_keep=1,
        ).logits[:, -1, :]
        cache = cached_forward(
            model, {"input_ids": prefix, "attention_mask": torch.ones((1, 700))}
        ).past_key_values
        actual = cached_forward(model, {
            "input_ids": XpuInput(suffix),
            "attention_mask": torch.ones((1, 2200)),
            "past_key_values": cache,
        }).logits[:, -1, :]
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)
