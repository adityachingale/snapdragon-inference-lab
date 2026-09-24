from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoImageProcessor


MODEL_ID = "microsoft/resnet-18"
IMAGE_PATH = Path("data/dog.jpg")
OUTPUT_DIR = Path("data/qnn_inputs")
OUTPUT_PATH = OUTPUT_DIR / "dog_pixel_values_nhwc.raw"


def main():
    # Load exactly the same Hugging Face processor used during
    # our PyTorch/ONNX validation.
    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID,
        use_fast=False,
    )

    # Load the same reference image.
    image = Image.open(IMAGE_PATH).convert("RGB")

    # Apply the model's expected preprocessing.
    # Hugging Face returns the ResNet tensor in NCHW layout:
    # [batch, channels, height, width] = [1, 3, 224, 224].
    inputs = processor(images=image, return_tensors="np")

    pixel_values_nchw = np.asarray(
        inputs["pixel_values"],
        dtype=np.float32,
        order="C",
    )

    assert pixel_values_nchw.shape == (1, 3, 224, 224), (
        f"Unexpected Hugging Face input shape: {pixel_values_nchw.shape}"
    )

    # QAIRT exposes the QNN model input as NHWC:
    # [batch, height, width, channels].
    # Convert NCHW -> NHWC.
    pixel_values_nhwc = np.transpose(
        pixel_values_nchw,
        (0, 2, 3, 1),
    )

    # qnn-net-run consumes raw bytes with no shape/layout metadata.
    # Make sure the NHWC tensor is contiguous float32 data.
    pixel_values_nhwc = np.ascontiguousarray(
        pixel_values_nhwc,
        dtype=np.float32,
    )

    assert pixel_values_nhwc.shape == (1, 224, 224, 3), (
        f"Unexpected QNN input shape: {pixel_values_nhwc.shape}"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write raw binary tensor values — no .npy header.
    pixel_values_nhwc.tofile(OUTPUT_PATH)

    print(f"Input image: {IMAGE_PATH}")
    print(f"Hugging Face tensor shape (NCHW): {pixel_values_nchw.shape}")
    print(f"QNN tensor shape (NHWC): {pixel_values_nhwc.shape}")
    print(f"Tensor dtype: {pixel_values_nhwc.dtype}")
    print(f"Min: {pixel_values_nhwc.min():.6f}")
    print(f"Max: {pixel_values_nhwc.max():.6f}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size:,} bytes")

    expected_size = 1 * 224 * 224 * 3 * np.dtype(np.float32).itemsize
    assert OUTPUT_PATH.stat().st_size == expected_size, (
        f"Unexpected QNN input file size: {OUTPUT_PATH.stat().st_size} bytes; "
        f"expected {expected_size} bytes"
    )


if __name__ == "__main__":
    main()