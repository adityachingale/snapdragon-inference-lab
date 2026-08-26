from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from pathlib import Path
import numpy as np
import onnxruntime as ort

MODEL_ID = "microsoft/resnet-18"
IMAGE_PATH = Path("data/dog.jpg")
ONNX_PATH = Path("models/resnet18/resnet18.onnx")

def main():
    
    # Pytorch Runtime Inference
    
    # Load the HuggingFace resnet model
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    
    model.eval()
    
    # Load the test image
    image = Image.open(IMAGE_PATH).convert("RGB")
    
    # Preprocess the image to convert into tensor.
    inputs = processor(images=image, return_tensors="pt")
    
    pixel_values = inputs["pixel_values"]
    
    print("")
    print("Input Shape: ", pixel_values.shape)
    print("Input Dtype: ", pixel_values.dtype)
    
    # Pytorch Inference
    with torch.inference_mode():
        pytorch_outputs = model(pixel_values=pixel_values)
    pytorch_logits = pytorch_outputs.logits
    
    print("\n PYTORCH Output:")
    print("Shape: ", pytorch_logits.shape)

    # =======================
    # ONNX Runtime Inference
    # =======================
    
    # Create an ONNX Runtime Inference Session
    ort_session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"],)
    
    onnx_input = pixel_values.numpy()
    
    # Run the ONNX model
    onnx_outputs = ort_session.run(None, {"pixel_values": onnx_input,},)
    
    # Normalize the runtime output to an ndarray before accessing its shape.
    onnx_logits = np.asarray(onnx_outputs[0])

    print("\nONNX Runtime output:")
    print("Shape:", onnx_logits.shape)
    
    # ========================================================
    # COMPARE PYTORCH VS ONNX
    # ========================================================

    # PyTorch logits are still a torch.Tensor.
    # Convert them to NumPy

    pytorch_logits_np = pytorch_logits.numpy()

    # Compute absolute difference element-by-element.
    # Example:
    # PyTorch:  [1.0000, 2.0000]
    # ONNX:     [1.0001, 1.9999]
    # diff:     [0.0001, 0.0001]

    absolute_difference = np.abs(pytorch_logits_np - onnx_logits)

    # Largest difference among all 1000 logits.

    max_difference = absolute_difference.max()

    # Average difference across all logits.

    mean_difference = absolute_difference.mean()

    print("\nNumerical comparison:")
    print(f"Max absolute difference:  {max_difference:.10f}")
    print(f"Mean absolute difference: {mean_difference:.10f}")



    outputs_match = np.allclose(pytorch_logits_np, onnx_logits, rtol=1e-4, atol=1e-5,)

    print("\nOutputs match within tolerance:", outputs_match)

    # ========================================================

    # COMPARE TOP-5 PREDICTIONS

    # ========================================================

    # PyTorch probabilities

    pytorch_probs = torch.softmax(pytorch_logits, dim=-1,)

    # Convert ONNX logits into a PyTorch tensor

    onnx_logits_tensor = torch.from_numpy(onnx_logits)

    onnx_probs = torch.softmax(onnx_logits_tensor,dim=-1,)

    pytorch_top5_probs, pytorch_top5_ids = torch.topk(pytorch_probs, k=5, dim=-1,)

    onnx_top5_probs, onnx_top5_ids = torch.topk(onnx_probs, k=5, dim=-1,)

    print("\nPyTorch top-5:")

    for probability, class_id in zip(pytorch_top5_probs[0], pytorch_top5_ids[0],):

        label = model.config.id2label[class_id.item()]

        print(f"{label:30s} {probability.item():.4%}")

    print("\nONNX Runtime top-5:")

    for probability, class_id in zip(onnx_top5_probs[0], onnx_top5_ids[0],):

        label = model.config.id2label[class_id.item()]

        print(f"{label:30s} {probability.item():.4%}")

if __name__ == "__main__":
    main()