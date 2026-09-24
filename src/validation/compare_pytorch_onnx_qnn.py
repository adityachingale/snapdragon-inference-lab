from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


MODEL_ID = "microsoft/resnet-18"

IMAGE_PATH = Path("data/dog.jpg")
ONNX_PATH = Path("models/resnet18/resnet18.onnx")
QNN_OUTPUT_PATH = Path(
    "data/qnn_outputs/resnet18_local/output/Result_0/logits.raw"
)


def compare(name_a, a, name_b, b):
    """Compare two 1000-element logit vectors."""
    diff = np.abs(a - b)

    print(f"\n{name_a} vs {name_b}")
    print("-" * 60)
    print(f"Max abs diff:  {diff.max():.10f}")
    print(f"Mean abs diff: {diff.mean():.10f}")
    print(f"RMSE:          {np.sqrt(np.mean((a - b) ** 2)):.10f}")
    print(f"Top-1 match:   {np.argmax(a) == np.argmax(b)}")


def print_top5(name, logits, id2label):
    probabilities = torch.softmax(
        torch.from_numpy(logits),
        dim=0,
    )

    top5 = torch.topk(probabilities, k=5)

    print(f"\n{name} Top-5")
    print("-" * 60)

    for rank, (idx, prob) in enumerate(
        zip(top5.indices.tolist(), top5.values.tolist()),
        start=1,
    ):
        print(
            f"{rank}. [{idx:4d}] "
            f"{id2label[idx]:35s} "
            f"{prob * 100:.4f}%"
        )


def main():
    # ---------------------------------------------------------
    # 1. Prepare exactly the same input used throughout testing
    # ---------------------------------------------------------

    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID,
        use_fast=False,
    )

    image = Image.open(IMAGE_PATH).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values"]

    print("Input shape:", tuple(pixel_values.shape))
    print("Input dtype:", pixel_values.dtype)

    # ---------------------------------------------------------
    # 2. PyTorch inference
    # ---------------------------------------------------------

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID
    )

    model.eval()

    with torch.no_grad():
        pytorch_logits = (
            model(pixel_values=pixel_values)
            .logits
            .cpu()
            .numpy()
            .squeeze()
            .astype(np.float32)
        )

    # ---------------------------------------------------------
    # 3. ONNX Runtime inference
    # ---------------------------------------------------------

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    onnx_logits = session.run(
        ["logits"],
        {
            "pixel_values": (
                pixel_values.cpu().numpy().astype(np.float32)
            )
        },
    )[0]

    onnx_logits = (
        np.asarray(onnx_logits)
        .squeeze()
        .astype(np.float32)
    )

    # ---------------------------------------------------------
    # 4. Physical S24 QNN result
    # ---------------------------------------------------------

    qnn_logits = np.fromfile(
        QNN_OUTPUT_PATH,
        dtype=np.float32,
    )

    # ---------------------------------------------------------
    # 5. Sanity checks
    # ---------------------------------------------------------

    assert pytorch_logits.shape == (1000,)
    assert onnx_logits.shape == (1000,)
    assert qnn_logits.shape == (1000,)

    print("\nOutput shapes")
    print("-" * 60)
    print("PyTorch:", pytorch_logits.shape)
    print("ONNX:   ", onnx_logits.shape)
    print("S24 QNN:", qnn_logits.shape)

    # ---------------------------------------------------------
    # 6. Numerical comparisons
    # ---------------------------------------------------------

    compare(
        "PyTorch",
        pytorch_logits,
        "ONNX Runtime",
        onnx_logits,
    )

    compare(
        "PyTorch",
        pytorch_logits,
        "S24 QNN",
        qnn_logits,
    )

    compare(
        "ONNX Runtime",
        onnx_logits,
        "S24 QNN",
        qnn_logits,
    )

    # ---------------------------------------------------------
    # 7. Semantic comparison
    # ---------------------------------------------------------

    id2label = model.config.id2label

    print_top5(
        "PyTorch",
        pytorch_logits,
        id2label,
    )

    print_top5(
        "ONNX Runtime",
        onnx_logits,
        id2label,
    )

    print_top5(
        "Physical S24 QNN",
        qnn_logits,
        id2label,
    )


if __name__ == "__main__":
    main()