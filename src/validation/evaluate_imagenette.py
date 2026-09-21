from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoImageProcessor, AutoConfig

MODEL_ID = "microsoft/resnet-18"
DATASET_DIR = Path("data/imagenette2/val")
BASELINE_MODEL = Path("models/resnet18/resnet18.onnx" )
W8A8_MODEL = Path("models/resnet18/resnet18_w8a8_package.onnx/model.onnx" )
W8A16_MODEL = Path("models/resnet18/resnet18_w8a16_package.onnx/model.onnx")

def build_wnid_to_class_id():
    """
    Hugging Face's ResNet config contains the ImageNet class labels.

    Imagenette directories use ImageNet WordNet IDs (WNIDs), such as:
        n01440764

    We therefore need to map each WNID to the corresponding one of the
    model's 1000 output classes.
    """
    config = AutoConfig.from_pretrained(MODEL_ID)
    # The Hugging Face config's id2label gives human-readable labels,
    # not necessarily WNIDs, so we'll explicitly map Imagenette's
    # 10 classes to their ImageNet-1K class indices.
    #
    # These are the standard Imagenette classes.
    wnid_to_class_id = {
        "n01440764": 0,    # tench
        "n02102040": 217,  # English springer
        "n02979186": 482,  # cassette player
        "n03000684": 491,  # chain saw
        "n03028079": 497,  # church
        "n03394916": 566,  # French horn
        "n03417042": 569,  # garbage truck
        "n03425413": 571,  # gas pump
        "n03445777": 574,  # golf ball
        "n03888257": 701,  # parachute
    }
    return wnid_to_class_id


def get_image_paths():
    """Return (image_path, ground_truth_class_id) pairs."""
    wnid_to_class_id = build_wnid_to_class_id()
    samples = []
    for wnid, class_id in wnid_to_class_id.items():
        class_dir = DATASET_DIR / wnid
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing Imagenette class directory: {class_dir}" )
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                samples.append((image_path, class_id) )
    return samples


def evaluate_model(model_path, samples, processor):
    """Evaluate one ONNX model on Imagenette."""
    print(f"\nLoading model: {model_path}")
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print(f"ONNX input: {input_name}")
    top1_correct = 0
    top5_correct = 0
    total = len(samples)
    for index, (image_path, true_class) in enumerate(samples, start=1):
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="np")
        pixel_values = inputs["pixel_values"].astype(np.float32 )
        outputs = session.run(None, {input_name: pixel_values})
        logits = np.asarray(outputs[0])[0]
        # Highest scoring class.
        top1 = int(np.argmax(logits))
        # Five highest scoring classes.
        top5 = np.argsort(logits)[-5:]
        if top1 == true_class:
            top1_correct += 1
        if true_class in top5:
            top5_correct += 1
        if index % 250 == 0:
            print(f"Processed {index}/{total} images..." )
    top1_accuracy = top1_correct / total
    top5_accuracy = top5_correct / total
    return {
        "total": total,
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
    }


def print_results(name, results):
    print(f"\n========== {name} ==========")
    print(f"Images: {results['total']}" )
    print(
        f"Top-1: "
        f"{results['top1_correct']}/{results['total']} "
        f"({results['top1_accuracy'] * 100:.2f}%)"
    )
    print(
        f"Top-5: "
        f"{results['top5_correct']}/{results['total']} "
        f"({results['top5_accuracy'] * 100:.2f}%)"
    )


def main():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=False)
    samples = get_image_paths()
    print(f"Found {len(samples)} Imagenette validation images." )
    baseline_results = evaluate_model(BASELINE_MODEL, samples, processor)
    print_results("BASELINE", baseline_results)
    w8a8_results = evaluate_model(W8A8_MODEL, samples, processor)
    print_results("W8A8", w8a8_results)
    w8a16_results = evaluate_model(W8A16_MODEL, samples, processor)
    print_results("W8A16", w8a16_results)
    print("\n========== ACCURACY DELTA For W8A8==========")
    top1_delta = (w8a8_results["top1_accuracy"] - baseline_results["top1_accuracy"] ) * 100
    top5_delta = (w8a8_results["top5_accuracy"] - baseline_results["top5_accuracy"] ) * 100
    print(f"Top-1 delta: {top1_delta:+.2f} percentage points" )
    print(f"Top-5 delta: {top5_delta:+.2f} percentage points" )
    print("\n========== ACCURACY DELTA For W8A16==========")
    top1_delta = (w8a16_results["top1_accuracy"] - baseline_results["top1_accuracy"] ) * 100
    top5_delta = (w8a16_results["top5_accuracy"] - baseline_results["top5_accuracy"] ) * 100
    print(f"Top-1 delta: {top1_delta:+.2f} percentage points" )
    print(f"Top-5 delta: {top5_delta:+.2f} percentage points" )
    
    print("\n========== ACCURACY DELTA For W8A16 vs W8A8==========")
    top1_delta = (w8a16_results["top1_accuracy"] - w8a8_results["top1_accuracy"] ) * 100
    top5_delta = (w8a16_results["top5_accuracy"] - w8a8_results["top5_accuracy"] ) * 100
    print(f"Top-1 delta: {top1_delta:+.2f} percentage points" )
    print(f"Top-5 delta: {top5_delta:+.2f} percentage points" )
    
    print("\n========== SUMMARY ==========")
    print(f"Baseline Top-1: {baseline_results['top1_accuracy'] * 100:.2f}%")
    print(f"W8A8 Top-1: {w8a8_results['top1_accuracy'] * 100:.2f}%")
    print(f"W8A16 Top-1: {w8a16_results['top1_accuracy'] * 100:.2f}%")
    


if __name__ == "__main__":
    main()
