import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from semif_phase1.artifacts import write_new_outputs


def benchmark(name):
    path = Path(__file__).resolve().parents[1] / "benchmarks" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_existing_predictions_preserved(tmp_path):
    report = tmp_path / "report.json"
    predictions = tmp_path / "report.predictions.jsonl"
    predictions.write_text("original evidence")
    with pytest.raises(FileExistsError):
        write_new_outputs({report: "new report", predictions: "replacement"})
    assert predictions.read_text() == "original evidence"
    assert not report.exists()


@pytest.mark.parametrize("name", ["shape777", "shape777_reranker"])
@pytest.mark.parametrize("collision", ["sidecar", "report"])
def test_benchmark_rejects_collisions_before_loading(monkeypatch, tmp_path, name, collision):
    report = tmp_path / "run.json"
    predictions = report.with_suffix(".predictions.jsonl")
    if collision == "sidecar":
        predictions.write_text("original")
    else:
        report.write_text("original report")
    monkeypatch.setattr(sys, "argv", [name, "--model", "unused", "--revision", "a" * 40,
                                     "--input", "absent.jsonl", "--output", str(report)])
    with pytest.raises(SystemExit) as error:
        benchmark(name).main()
    assert error.value.code == 2
    if collision == "sidecar":
        assert predictions.read_text() == "original"


@pytest.mark.parametrize("answer", ['[{}]', '[[]]', '[null]', '["yes"]'])
def test_generation_records_invalid_array_members(answer):
    import torch

    class Inputs(dict):
        def to(self, device):
            return self

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

        def __call__(self, *args, **kwargs):
            return Inputs(input_ids=torch.tensor([[1]]))

        def decode(self, *args, **kwargs):
            return answer

    model = SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]),
                            generate=lambda **kwargs: torch.tensor([[1, 2]]))
    result = benchmark("decision_vs_generation").run_generation(
        model, Tokenizer(), "state", [{"question": "criterion"}], 5)
    assert result["valid_complete_array"] == (answer == '["yes"]')


def test_generation_prefill_chunking_matches_plain_generate():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
    )).eval()
    ids = torch.randint(0, 32, (1, 25))
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        plain = model.generate(
            input_ids=ids, attention_mask=mask, do_sample=False,
            max_new_tokens=8, use_cache=True,
        )
        chunked = model.generate(
            input_ids=ids, attention_mask=mask, do_sample=False,
            max_new_tokens=8, use_cache=True, prefill_chunk_size=7,
        )
    assert torch.equal(plain, chunked)
