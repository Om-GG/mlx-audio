import glob
import importlib
import json
import logging
import shutil
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional

import mlx.core as mx
import numpy as np
import soundfile as sf
from huggingface_hub import snapshot_download
from scipy import signal
from mlx.utils import tree_flatten
from mlx_lm.convert import mixed_quant_predicate_builder
from mlx_lm.utils import dequantize_model, quantize_model, save_config, save_model

SAMPLE_RATE = 16000

MODEL_REMAPPING = {}
MAX_FILE_SIZE_GB = 5
MODEL_CONVERSION_DTYPES = ["float16", "bfloat16", "float32"]


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    resampled = signal.resample_poly(audio, up, down, padtype="edge")
    return resampled


def load_audio(
    file: str = Optional[str],
    sr: int = SAMPLE_RATE,
    from_stdin=False,
    dtype: mx.Dtype = mx.float32,
):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    audio, sample_rate = sf.read(file, always_2d=True)
    if sample_rate != sr:
        audio = resample_audio(audio, sample_rate, sr)
    return mx.array(audio, dtype=dtype).mean(axis=1)


def get_model_path(
    path_or_hf_repo: str, revision: Optional[str] = None, force_download: bool = False
) -> Path:
    """
    Ensures the model is available locally. If the path does not exist locally,
    it is downloaded from the Hugging Face Hub.

    Args:
        path_or_hf_repo (str): The local path or Hugging Face repository ID of the model.
        revision (str, optional): A revision id which can be a branch name, a tag, or a commit hash.

    Returns:
        Path: The path to the model.
    """
    model_path = Path(path_or_hf_repo)

    if not model_path.exists():
        model_path = Path(
            snapshot_download(
                path_or_hf_repo,
                revision=revision,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.py",
                    "*.model",
                    "*.tiktoken",
                    "*.txt",
                    "*.jsonl",
                    "*.yaml",
                ],
                force_download=force_download,
            )
        )

    return model_path


# Get a list of all available model types from the models directory
def get_available_models():
    """
    Get a list of all available TTS model types by scanning the models directory.

    Returns:
        List[str]: A list of available model type names
    """
    models_dir = Path(__file__).parent / "models"
    available_models = []

    if models_dir.exists() and models_dir.is_dir():
        for item in models_dir.iterdir():
            if item.is_dir() and not item.name.startswith("__"):
                available_models.append(item.name)

    return available_models


def get_model_and_args(model_type: str, model_name: List[str]):
    """
    Retrieve the model architecture module based on the model type and name.

    This function attempts to find the appropriate model architecture by:
    1. Checking if the model_type is directly in the MODEL_REMAPPING dictionary
    2. Looking for partial matches in segments of the model_name

    Args:
        model_type (str): The type of model to load (e.g., "outetts").
        model_name (List[str]): List of model name components that might contain
                               remapping information.

    Returns:
        Tuple[module, str]: A tuple containing:
            - The imported architecture module
            - The resolved model_type string after remapping

    Raises:
        ValueError: If the model type is not supported (module import fails).
    """
    # Stage 1: Check if the model type is in the remapping
    model_type = MODEL_REMAPPING.get(model_type, model_type)

    # Stage 2: Check for partial matches in segments of the model name
    models = get_available_models()
    if model_name is not None:
        for part in model_name:
            # First check if the part matches an available model directory name
            if part in models:
                model_type = part

            # Then check if the part is in our custom remapping dictionary
            if part in MODEL_REMAPPING:
                model_type = MODEL_REMAPPING[part]
                break

    try:
        arch = importlib.import_module(f"mlx_audio.stt.models.{model_type}")
    except ImportError:
        msg = f"Model type {model_type} not supported."
        logging.error(msg)
        raise ValueError(msg)

    return arch, model_type


def load_model(model_path: str, lazy: bool = False, strict: bool = True, **kwargs):
    """
    Load and initialize the model from a given path.

    Args:
        model_path (str): The path to load the model from.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``

    Returns:
        nn.Module: The loaded and initialized model.

    Raises:
        FileNotFoundError: If the weight files (.safetensors) are not found.
        ValueError: If the model class or args class are not found or cannot be instantiated.
    """
    model_name = None
    model_type = None
    if isinstance(model_path, str):
        model_name = model_path.lower().split("/")[-1].split("-")
    elif isinstance(model_path, Path):
        index = model_path.parts.index("hub")
        model_name = model_path.parts[index + 1].lower().split("--")[-1].split("-")
    else:
        raise ValueError(f"Invalid model path type: {type(model_path)}")

    model_class, model_type = get_model_and_args(
        model_type=model_type, model_name=model_name
    )
    model = model_class.Model.from_pretrained(model_path)

    if not lazy:
        model.eval()

    return model


def upload_to_hub(path: str, upload_repo: str, hf_path: str):
    """Upload a converted model directory to the Hugging Face Hub."""

    import os

    from huggingface_hub import HfApi, ModelCard, logging

    from ..version import __version__

    card = ModelCard.load(hf_path)
    card.data.tags = ["mlx"] if card.data.tags is None else card.data.tags + ["mlx"]
    card.text = dedent(
        f"""
        # {upload_repo}
        This model was converted to MLX format from [`{hf_path}`](https://huggingface.co/{hf_path}) using mlx-audio version **{__version__}**.
        Refer to the [original model card](https://huggingface.co/{hf_path}) for more details on the model.
        ## Use with mlx

        ```bash
        pip install -U mlx-audio
        ```

        ```bash
        python -m mlx_audio.stt.generate --model {upload_repo} --audio YOUR_AUDIO.wav --output transcript
        ```
        """
    )
    card.save(os.path.join(path, "README.md"))

    logging.set_verbosity_info()

    api = HfApi()
    api.create_repo(repo_id=upload_repo, exist_ok=True)
    api.upload_folder(
        folder_path=path,
        repo_id=upload_repo,
        repo_type="model",
    )
    print(f"Upload successful, go to https://huggingface.co/{upload_repo} for details.")


def _load_voxtral_weights(model_path: Path) -> Dict[str, mx.array]:
    weight_files = sorted(model_path.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    weights: Dict[str, mx.array] = {}
    for file in weight_files:
        weights.update(mx.load(file))

    return weights


def convert(
    hf_path: str,
    mlx_path: str = "mlx_model",
    quantize: bool = False,
    q_group_size: int = 64,
    q_bits: int = 4,
    dtype: Optional[str] = None,
    upload_repo: Optional[str] = None,
    revision: Optional[str] = None,
    dequantize: bool = False,
    quant_predicate: Optional[str] = None,
):
    from .models.voxtral import Model, ModelConfig

    print("[INFO] Loading")
    model_path = get_model_path(hf_path, revision=revision)

    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config: Dict[str, Any] = json.load(f)

    model_config = ModelConfig.from_dict(config)
    model = Model(model_config)

    weights = model.sanitize(_load_voxtral_weights(model_path))
    model.load_weights(list(weights.items()))
    weights = dict(tree_flatten(model.parameters()))

    if dtype is None:
        dtype = config.get("torch_dtype", None)
    if dtype in MODEL_CONVERSION_DTYPES:
        print("[INFO] Using dtype:", dtype)
        target_dtype = getattr(mx, dtype)
        weights = {k: v.astype(target_dtype) for k, v in weights.items()}
        model.load_weights(list(weights.items()))
        config["torch_dtype"] = dtype

    if quantize and dequantize:
        raise ValueError("Choose either quantize or dequantize, not both.")

    predicate_from_model = getattr(
        model, "model_quant_predicate", lambda p, m, cfg: True
    )

    recipe_predicate = None
    if isinstance(quant_predicate, str):
        recipe_predicate = mixed_quant_predicate_builder(
            quant_predicate, model.language_model.model
        )

    def final_quant_predicate(path, module, cfg):
        if not hasattr(module, "to_quantized"):
            return False
        if hasattr(module, "weight") and module.weight.shape[-1] % q_group_size != 0:
            return False
        if not predicate_from_model(path, module, cfg):
            return False
        if recipe_predicate is None:
            return True
        inner_path = path.split("language_model.", 1)[-1]
        return recipe_predicate(inner_path, module, cfg)

    if quantize:
        print("[INFO] Quantizing")
        weights, config = quantize_model(
            model,
            config,
            q_group_size,
            q_bits,
            quant_predicate=final_quant_predicate,
        )

    if dequantize:
        print("[INFO] Dequantizing")
        model = dequantize_model(model)
        weights = dict(tree_flatten(model.parameters()))
        config.pop("quantization", None)

    if isinstance(mlx_path, str):
        mlx_path = Path(mlx_path)

    mlx_path.mkdir(parents=True, exist_ok=True)

    copy_patterns = [
        "*.py",
        "*.json",
        "*.model",
        "*.tiktoken",
        "*.txt",
        "*.yaml",
        "*.safetensors",
    ]

    for pattern in copy_patterns:
        for file in glob.glob(str(model_path / pattern)):
            shutil.copy(file, mlx_path)

        for file in glob.glob(str(model_path / "**" / pattern), recursive=True):
            rel_path = Path(file).relative_to(model_path)
            dest_dir = mlx_path / rel_path.parent
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(file, dest_dir)

    save_model(mlx_path, model, donate_model=True)
    save_config(config, config_path=mlx_path / "config.json")

    if upload_repo is not None:
        upload_to_hub(str(mlx_path), upload_repo, hf_path)
