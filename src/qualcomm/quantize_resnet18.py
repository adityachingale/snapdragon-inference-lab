from pathlib import Path
import qai_hub as hub
from src.data.prepare_calibration import prepare_calibration_data


ONNX_PATH = Path("models/resnet18/resnet18_package.onnx")
W8A8_OUTPUT = Path("models/resnet18/resnet18_w8a8.onnx")
W8A16_OUTPUT = Path("models/resnet18/resnet18_w8a16.onnx")


def quantize_model(
    client,
    model,
    calibration_data,
    weights_dtype,
    activations_dtype,
    name,
    output_path,
):
    print(f"\nSubmitting quantization job: {name}")

    quantize_job = client.submit_quantize_job(
        model=model,
        calibration_data=calibration_data,
        weights_dtype=weights_dtype,
        activations_dtype=activations_dtype,
        name=name,
    )

    print(f"Job submitted: {quantize_job.url}")

    status = quantize_job.wait()

    if not status.success:
        raise RuntimeError(
            f"Quantization failed: {status}. "
            f"Job: {quantize_job.url}"
        )

    quantized_model = quantize_job.get_target_model()

    if quantized_model is None:
        raise RuntimeError(
            f"Quantization produced no model: {quantize_job.url}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    quantized_model.download(str(output_path))

    print(f"Quantized model downloaded to: {output_path}")

    return quantized_model


def main():

    client = hub.Client()

    # ---------------------------------------------------------
    # Prepare our 100 representative calibration images.
    # ---------------------------------------------------------

    calibration_data = prepare_calibration_data()

    print(
        f"Calibration samples: "
        f"{len(calibration_data['pixel_values'])}"
    )

    # ---------------------------------------------------------
    # Upload the original unquantized ONNX model.
    # ---------------------------------------------------------

    print("\nUploading source ONNX model...")

    source_model = client.upload_model(str(ONNX_PATH))

    # ---------------------------------------------------------
    # Experiment 1: W8A8
    #
    # W8 = INT8 weights
    # A8 = INT8 activations
    # ---------------------------------------------------------

    quantize_model(
        client=client,
        model=source_model,
        calibration_data=calibration_data,
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT8,
        name="resnet18_w8a8",
        output_path=W8A8_OUTPUT,
    )

    # ---------------------------------------------------------
    # Experiment 2: W8A16
    #
    # W8  = INT8 weights
    # A16 = INT16 activations
    # ---------------------------------------------------------

    quantize_model(
        client=client,
        model=source_model,
        calibration_data=calibration_data,
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT16,
        name="resnet18_w8a16",
        output_path=W8A16_OUTPUT,
    )


if __name__ == "__main__":
    main()