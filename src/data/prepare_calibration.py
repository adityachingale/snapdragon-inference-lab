from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoImageProcessor


MODEL_ID = "microsoft/resnet-18"
CALIBRATION_DIR = Path("data/calibration/imagenette_samples")


def prepare_calibration_data():
    # Same preprocessing configuration used by our reference ResNet-18.
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)

    # Qualcomm's sample may contain nested directories, so search recursively.
    image_paths = sorted([path
            for path in CALIBRATION_DIR.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    if not image_paths:
        raise RuntimeError(
            f"No calibration images found in {CALIBRATION_DIR}"
        )

    print(f"Found {len(image_paths)} calibration images.")

    calibration_samples = []

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")

        inputs = processor(
            images=image,
            return_tensors="np",
        )

        pixel_values = inputs["pixel_values"]

        # Verify the tensor matches our ONNX input.
        if pixel_values.shape != (1, 3, 224, 224):
            raise ValueError(
                f"Unexpected shape for {image_path}: "
                f"{pixel_values.shape}"
            )

        if pixel_values.dtype != np.float32:
            pixel_values = pixel_values.astype(np.float32)

        calibration_samples.append(pixel_values)

    calibration_data = {
        "pixel_values": calibration_samples
    }

    return calibration_data


if __name__ == "__main__":
    calibration_data = prepare_calibration_data()

    samples = calibration_data["pixel_values"]

    print("\nCalibration data prepared successfully.")
    print(f"Number of samples: {len(samples)}")
    print(f"First sample shape: {samples[0].shape}")
    print(f"First sample dtype: {samples[0].dtype}")
    print(f"First sample min: {samples[0].min():.4f}")
    print(f"First sample max: {samples[0].max():.4f}")