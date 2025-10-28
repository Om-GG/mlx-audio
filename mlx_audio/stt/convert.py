"""Command-line entry point for converting STT models to MLX format."""

import argparse

from .utils import MODEL_CONVERSION_DTYPES, convert

QUANT_RECIPES = ["mixed_2_6", "mixed_3_4", "mixed_3_6", "mixed_4_6"]


def configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face STT model to MLX format"
    )

    parser.add_argument("--hf-path", type=str, help="Path to the Hugging Face model.")
    parser.add_argument(
        "--mlx-path", type=str, default="mlx_model", help="Path to save the MLX model."
    )
    parser.add_argument(
        "-q", "--quantize", help="Generate a quantized model.", action="store_true"
    )
    parser.add_argument(
        "--q-group-size", help="Group size for quantization.", type=int, default=64
    )
    parser.add_argument(
        "--q-bits", help="Bits per weight for quantization.", type=int, default=4
    )
    parser.add_argument(
        "--quant-predicate",
        help="Mixed-bit quantization recipe.",
        choices=QUANT_RECIPES,
        type=str,
        required=False,
    )
    parser.add_argument(
        "--dtype",
        help="Type to save the parameters, ignored if -q is given.",
        type=str,
        choices=MODEL_CONVERSION_DTYPES,
        default="float16",
    )
    parser.add_argument(
        "--revision",
        help="Specific revision of the source repo to download.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--upload-repo",
        help="The Hugging Face repo to upload the model to.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-d",
        "--dequantize",
        help="Dequantize a quantized model.",
        action="store_true",
        default=False,
    )
    return parser


def main():
    parser = configure_parser()
    args = parser.parse_args()
    convert(**vars(args))


if __name__ == "__main__":
    main()
