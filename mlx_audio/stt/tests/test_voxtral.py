import json
import types
from contextlib import contextmanager
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

mx = pytest.importorskip(
    "mlx.core", reason="MLX runtime required", exc_type=ImportError
)
pytest.importorskip(
    "mlx.nn", reason="MLX runtime required", exc_type=ImportError
)

import mlx_audio.stt.models.voxtral.voxtral as voxtral_module
from mlx_audio.stt.models.voxtral.voxtral import Model, STTOutput
from mlx_audio.stt.utils import convert, load_model


class DummyEmbedTokens:
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size

    def __call__(self, input_ids):
        ids = np.array(input_ids)
        shape = tuple(ids.shape) + (self.hidden_size,)
        return mx.zeros(shape)

    def as_linear(self, values):
        return values


class DummyLlamaModel:
    def __init__(self, config):
        self.config = config
        self.layers: List[object] = []
        self.embed_tokens = DummyEmbedTokens(config.hidden_size)

    def __call__(self, inputs=None, mask=None, cache=None, input_embeddings=None):
        if input_embeddings is not None:
            return input_embeddings
        if inputs is None:
            return mx.zeros((1, 1, self.config.hidden_size))
        return inputs

    def model_quant_predicate(self, path, module, config):
        return True


def fake_generate_step(prompt, input_embeddings, model, max_tokens, sampler):
    yield 10, mx.array([0.0])
    yield 2, mx.array([0.0])


def fake_make_sampler(*args, **kwargs):
    def sampler(logits):
        return logits

    return sampler


class DummyAutoProcessor:
    def __init__(self, num_mel_bins: int, audio_token_id: int):
        self._num_mel_bins = num_mel_bins
        self._audio_token_id = audio_token_id
        tokenizer = MagicMock()
        tokenizer.eos_token_ids = [2]
        tokenizer.encode.return_value = [audio_token_id]
        tokenizer.decode.side_effect = lambda tokens: "token:" + ",".join(map(str, tokens))
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        config_path = Path(path) / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        num_mel_bins = config["audio_config"]["num_mel_bins"]
        audio_token_id = config.get("audio_token_id", 0)
        return cls(num_mel_bins=num_mel_bins, audio_token_id=audio_token_id)

    def apply_transcription_request(self, *, language, audio, model_id):
        input_ids = [[self._audio_token_id, 1]]
        input_features = np.zeros((1, 1, self._num_mel_bins), dtype=np.float32)
        return {
            "input_ids": input_ids,
            "input_features": input_features,
        }

    def decode(self, tokens):
        return "decoded:" + ",".join(map(str, tokens))


@pytest.fixture(autouse=True)
def stub_mlx_lm_modules(monkeypatch):
    mlx_lm_module = types.ModuleType("mlx_lm")
    mlx_lm_module.__path__ = []

    models_module = types.ModuleType("mlx_lm.models")
    models_module.__path__ = []
    llama_module = types.ModuleType("mlx_lm.models.llama")
    llama_module.LlamaModel = DummyLlamaModel

    generate_module = types.ModuleType("mlx_lm.generate")
    generate_module.generate_step = fake_generate_step

    sample_utils_module = types.ModuleType("mlx_lm.sample_utils")
    sample_utils_module.make_sampler = fake_make_sampler

    utils_module = types.ModuleType("mlx_lm.utils")
    utils_module.quantize_model = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("quantize_model should not be called in tests")
    )
    utils_module.dequantize_model = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("dequantize_model should not be called in tests")
    )
    utils_module.save_model = lambda *args, **kwargs: None
    utils_module.save_config = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm_module)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", models_module)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.llama", llama_module)
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", generate_module)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils_module)
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", utils_module)

    setattr(mlx_lm_module, "models", models_module)
    setattr(mlx_lm_module, "generate", generate_module)
    setattr(mlx_lm_module, "sample_utils", sample_utils_module)
    setattr(models_module, "llama", llama_module)


@pytest.fixture(autouse=True)
def stub_wired_limit(monkeypatch):
    @contextmanager
    def noop_wired_limit(model, streams=None):
        yield

    monkeypatch.setattr(
        "mlx_audio.stt.models.voxtral.voxtral.wired_limit", noop_wired_limit
    )


@pytest.fixture
def capture_load_weights(monkeypatch):
    calls = []

    def fake_load_weights(self, items, strict=True):
        calls.append((self, items, strict))
        self._loaded_weights = dict(items)
        return self

    monkeypatch.setattr(Model, "load_weights", fake_load_weights)
    return calls


@pytest.fixture(autouse=True)
def patch_auto_processor(monkeypatch):
    monkeypatch.setattr(
        "mlx_audio.stt.models.voxtral.voxtral.AutoProcessor", DummyAutoProcessor
    )


def test_from_pretrained_applies_quantization_and_generates(
    tmp_path, monkeypatch, capture_load_weights
):
    config = {
        "audio_config": {
            "d_model": 16,
            "encoder_layers": 1,
            "encoder_attention_heads": 1,
            "encoder_ffn_dim": 32,
            "num_mel_bins": 4,
            "intermediate_size": 8,
            "max_source_positions": 2,
        },
        "text_config": {
            "model_type": "llama",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "tie_word_embeddings": True,
        },
        "audio_token_id": 5,
        "quantization": {"group_size": 128, "bits": 4},
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    dummy_weight_path = tmp_path / "model-00001-of-00001.safetensors"
    dummy_weights = {"language_model.layers.0.weight": mx.zeros((1, 1))}

    monkeypatch.setattr(
        "mlx_audio.stt.models.voxtral.voxtral.glob.glob",
        lambda pattern: [str(dummy_weight_path)],
    )
    monkeypatch.setattr(
        "mlx_audio.stt.models.voxtral.voxtral.mx.load", lambda _: dummy_weights
    )
    monkeypatch.setattr(voxtral_module.nn, "quantize", MagicMock())
    quantize_mock = voxtral_module.nn.quantize

    model = Model.from_pretrained(str(tmp_path))

    assert isinstance(model, Model)
    assert len(capture_load_weights) == 1
    assert model._loaded_weights == dummy_weights

    quantize_mock.assert_called_once()
    kwargs = quantize_mock.call_args.kwargs
    assert kwargs["group_size"] == 128
    assert kwargs["bits"] == 4
    predicate = kwargs["class_predicate"]

    spy = MagicMock(return_value=True)
    monkeypatch.setattr(model, "model_quant_predicate", spy)
    assert predicate("layers.0", object()) is True
    spy.assert_called_once()
    called_path, called_module, called_config = spy.call_args[0]
    assert called_path == "language_model.layers.0"
    assert called_config == model.config

    output = model.generate([mx.zeros((4,), dtype=mx.float32)])
    assert isinstance(output, STTOutput)
    assert output.text.startswith("decoded")


@pytest.fixture
def conversion_save_helpers(monkeypatch):
    saved = {}

    def fake_save_model(path, model, donate_model=True):
        weights = getattr(model, "_loaded_weights", {})
        target = Path(path) / "model-00001-of-00001.safetensors"
        mx.save(target, weights if weights else {"dummy": mx.zeros((1,))})
        saved["model_path"] = target

    def fake_save_config(config, config_path):
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)
        saved["config_path"] = config_path

    monkeypatch.setattr("mlx_lm.utils.save_model", fake_save_model)
    monkeypatch.setattr("mlx_lm.utils.save_config", fake_save_config)
    return saved


def test_conversion_pipeline_smoke(tmp_path, monkeypatch, capture_load_weights, conversion_save_helpers):
    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    config = {
        "audio_config": {
            "d_model": 16,
            "encoder_layers": 1,
            "encoder_attention_heads": 1,
            "encoder_ffn_dim": 32,
            "num_mel_bins": 4,
            "intermediate_size": 8,
            "max_source_positions": 2,
        },
        "text_config": {
            "model_type": "llama",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "tie_word_embeddings": True,
        },
        "audio_token_id": 5,
    }
    (hf_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    weights = {"language_model.layers.0.weight": mx.zeros((1, 1))}

    monkeypatch.setattr("mlx_audio.stt.utils.get_model_path", lambda *args, **kwargs: hf_dir)
    monkeypatch.setattr("mlx_audio.stt.utils._load_voxtral_weights", lambda _: weights)

    output_dir = tmp_path / "mlx"
    convert(hf_path=str(hf_dir), mlx_path=str(output_dir))

    assert (output_dir / "model-00001-of-00001.safetensors").exists()
    assert (output_dir / "config.json").exists()

    model = load_model(str(output_dir))
    assert isinstance(model, Model)
    assert len(capture_load_weights) >= 1

    result = model.generate([mx.zeros((4,), dtype=mx.float32)])
    assert isinstance(result, STTOutput)
    assert result.text.startswith("decoded")
