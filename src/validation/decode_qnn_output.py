import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

OUTPUT = "data/qnn_outputs/resnet18_local/output/Result_0/logits.raw"
MODEL_ID = "microsoft/resnet-18"

# QNN output: 1000 FP32 ImageNet logits
logits = np.fromfile(OUTPUT, dtype=np.float32)

print("Shape:", logits.shape)
print("dtype:", logits.dtype)
print("Min/Max:", logits.min(), logits.max())

assert logits.size == 1000, f"Expected 1000 logits, got {logits.size}"

# Convert logits to probabilities
probs = torch.softmax(torch.from_numpy(logits), dim=0)

# Get ImageNet label mapping from the HF model config
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
id2label = model.config.id2label

top5 = torch.topk(probs, k=5)

print("\nPhysical S24 QNN Top-5")
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