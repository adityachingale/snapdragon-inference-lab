# ResNet-18 — Snapdragon Benchmark

## Objective

Establish a baseline for progressively testing model architectures and quantization strategies on Qualcomm Snapdragon hardware. Results below record the completed experiment; documentation creation did not rerun compilation or profiling.

## Model

- Model ID: `microsoft/resnet-18`
- Task: ImageNet-1K classification
- Input: `1x3x224x224`
- Output: `1x1000` logits

## PyTorch Reference

[`src/reference/resnet18_pytorch.py`](../../src/reference/resnet18_pytorch.py) loads the Hugging Face image processor and pretrained model, preprocesses `data/dog.jpg` as RGB, and runs PyTorch inference in evaluation mode with gradients disabled. Softmax probabilities are used to display the top-5 predictions.

## ONNX Export

[`src/export/resnet18_to_onnx.py`](../../src/export/resnet18_to_onnx.py) exports the PyTorch model to ONNX with a float32 example input and opset 18, then checks the ONNX graph. The exporter produced external tensor storage:

- `resnet18.onnx`: model graph
- `resnet18.onnx.data`: external tensor weights

These generated artifacts remain ignored by Git.

## ONNX Validation

[`src/validation/compare_pytorch_onnx.py`](../../src/validation/compare_pytorch_onnx.py) compares PyTorch and ONNX Runtime CPU outputs using the same preprocessed test image.

| Check | Result |
| --- | --- |
| Input shape | `[1, 3, 224, 224]` |
| Output shape | `[1, 1000]` |
| Max absolute difference | 0.0000114441 |
| Mean absolute difference | 0.0000017367 |
| Outputs matched within tolerance | Yes (`rtol=1e-4`, `atol=1e-5`) |
| Top-5 predictions matched | Yes |

This validates numerical agreement of the ONNX conversion for the tested input. It is not a dataset-level accuracy measurement and does not validate the compiled QNN model's outputs.

## Qualcomm Compilation

[`src/qualcomm/compile_resnet18.py`](../../src/qualcomm/compile_resnet18.py) packages the external-data ONNX model with both the `.onnx` graph and `.onnx.data` weights in `resnet18_package.onnx/` for Qualcomm AI Hub.

| Property | Value |
| --- | --- |
| Device | Samsung Galaxy S24 (Family) |
| Chipset | Qualcomm Snapdragon 8 Gen 3 |
| SoC | SM8650 |
| Hexagon | v75 |
| Framework/runtime | QNN |
| HTP FP16 support | true |
| Compiled format | QNN DLC |
| Compilation | Successful |

The compilation options specify only `--target_runtime qnn_dlc`; profiling supplies no additional precision options. No explicit quantization is requested. Compiled weight and activation precision remain `TBD`: HTP FP16 support alone does not establish the precision used by the compiled model.

## Baseline Profiling

- Current pre-explicit-quantization floating-point baseline
- Estimated inference latency: **0.707 ms**

The script reads Qualcomm AI Hub's `execution_summary.estimated_inference_time` and converts microseconds to milliseconds. This measures estimated model inference latency, not the complete application/camera pipeline; no FPS claim is derived.

Model size, memory, dataset-level accuracy, throughput, and NPU utilization have not been measured. The shared benchmark CSV leaves model-size and accuracy fields blank; precision fields use `TBD` until verified.

## Quantization Plan

| Experiment | Definition | Status |
| --- | --- | --- |
| W8A16 | INT8 weights + INT16 activations | TBD |
| W8A8 | INT8 weights + INT8 activations | TBD |

Compare latency, model size, memory where available, dataset-level accuracy, and hardware/runtime behavior against the baseline. These are planned experiments; no quantized results have been collected.

## Future Model Ladder

ResNet-18 → MobileNet → object detection/YOLO → small Transformer → embedding model → small LLM → on-device RAG

The benchmark CSV uses task, input shape, precision, and named accuracy-metric fields to support future architectures. Leave uncollected measurements blank or `TBD` and document experiment-specific conditions in the notes.

## Experiment History

| Model | Experiment | Format | Estimated inference latency | Status |
| --- | --- | --- | --- | --- |
| ResNet-18 | baseline | QNN DLC | 0.707 ms | successful |
