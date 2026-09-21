# ResNet-18 — Snapdragon Benchmark

## Objective

Establish a baseline for progressively testing model architectures and quantization strategies on Qualcomm Snapdragon hardware. Results below record the completed experiment; this documentation update did not rerun evaluation, compilation, or profiling. Measurements and runtime observations are from the completed experiments reported by the project author.

## Model

- Model ID: `microsoft/resnet-18`
- Task: ImageNet-1K classification
- Input: `1x3x224x224`, float32
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
| OS | Android 14 |
| Chipset | Qualcomm Snapdragon 8 Gen 3 |
| SoC | SM8650 |
| Hexagon | v75 |
| Framework/runtime | QNN |
| HTP FP16 support | true |
| Compiled format | QNN DLC |
| Compilation | Successful |

For the baseline, the compilation options specify only `--target_runtime qnn_dlc`; profiling supplies no additional precision options. No explicit quantization is requested. Compiled weight and activation precision remain `TBD`: HTP FP16 support alone does not establish the precision used by the compiled model.

## Baseline Profiling

- Current pre-explicit-quantization floating-point baseline
- Estimated inference latency: **0.707 ms**

The script reads Qualcomm AI Hub's `execution_summary.estimated_inference_time` and converts microseconds to milliseconds. This measures estimated model inference latency, not the complete application/camera pipeline; no FPS claim is derived.

## Calibration and Evaluation

W8A8 and W8A16 each used 100 representative Imagenette calibration images, separate from the 3,925 labeled images in the full-size Imagenette validation split. Calibration determines quantization parameters; validation measures the resulting accuracy.

All configurations used `AutoImageProcessor` from `microsoft/resnet-18` with the same slow preprocessing behavior. The evaluator explicitly sets `use_fast=False`; the calibration helper uses the equivalent default behavior of the experiment environment. Preserve this processor behavior when reproducing the run.

[`src/validation/evaluate_imagenette.py`](../../src/validation/evaluate_imagenette.py) evaluates ONNX models with ONNX Runtime's `CPUExecutionProvider`. Imagenette covers 10 ImageNet classes, mapped from directory WNIDs to the model's 1,000 output indices; predictions are ranked across all 1,000 logits. These are **Imagenette validation accuracy** measurements, not full ImageNet validation accuracy or direct accuracy measurements of the DLC on the NPU.

## Quantization Experiments

[`src/qualcomm/quantize_resnet18.py`](../../src/qualcomm/quantize_resnet18.py) uploads the original ONNX package and submits Qualcomm AI Hub quantization jobs using calibration data from [`src/data/prepare_calibration.py`](../../src/data/prepare_calibration.py).

| Configuration | Requested weights | Requested activations | Flow |
| --- | --- | --- | --- |
| W8A8 | INT8 | INT8 | QDQ ONNX → QNN DLC |
| W8A16 | INT8 | INT16 | QDQ ONNX → QNN DLC |

QDQ denotes QuantizeLinear/DequantizeLinear operations representing quantization in ONNX. The packaged quantized models are compiled with `--target_runtime qnn_dlc` by [`compile_resnet18_w8a8.py`](../../src/qualcomm/compile_resnet18_w8a8.py) and [`compile_resnet18_w8a16.py`](../../src/qualcomm/compile_resnet18_w8a16.py). Each script waits for compilation, downloads the DLC, submits the target model for profiling, and retrieves the profile.

Profiling confirmed NPU execution for the baseline, W8A8, and W8A16. Observed runtime boundary conversions were:

| Configuration | Input conversion | Output conversion |
| --- | --- | --- |
| W8A8 | FLOAT32 → UFIXED_POINT_8 | UFIXED_POINT_8 → FLOAT32 |
| W8A16 | FLOAT32 → UFIXED_POINT_16 | UFIXED_POINT_16 → FLOAT32 |

INT8/INT16 describe the requested quantization settings; the unsigned fixed-point labels above describe the observed QNN runtime tensors.

## Benchmark Summary

Accuracy uses 3,925 Imagenette validation images; latency is Qualcomm AI Hub estimated NPU inference latency. DLC sizes are approximate downloaded artifact sizes, not runtime memory measurements.

| Configuration | Weights | Activations | Top-1 | Top-5 | Latency | DLC size |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | Floating* | Floating* | 81.45% | 95.97% | 0.707 ms | ~45 MB |
| W8A8 | INT8 | INT8 | 81.89% | 96.10% | 0.310 ms | ~12 MB |
| W8A16 | INT8 | INT16 | 81.48% | 95.92% | 0.377 ms | ~11.5 MB |

*Floating denotes the pre-explicit-quantization baseline. Actual compiled precision remains unverified; this is not a claim of strictly FP32 NPU execution, because QNN may internally optimize precision.

| Configuration | Top-1 correct / total | Top-5 correct / total |
| --- | --- | --- |
| Baseline | 3197 / 3925 | 3767 / 3925 |
| W8A8 | 3214 / 3925 | 3772 / 3925 |
| W8A16 | 3198 / 3925 | 3765 / 3925 |

| Comparison | Top-1 delta | Top-5 delta | Latency comparison |
| --- | --- | --- | --- |
| W8A8 vs baseline | +0.43 pp | +0.13 pp | 56.2% lower; 2.28× inference speedup |
| W8A16 vs baseline | +0.03 pp | -0.05 pp | 46.7% lower; 1.88× inference speedup |
| W8A16 vs W8A8 | -0.41 pp | -0.18 pp | 21.6% higher |

Here `pp` means percentage points. Accuracy deltas use raw correct counts divided by 3,925 before rounding. Latency reduction is `(reference - candidate) / reference × 100`; speedup is `reference / candidate`. W8A8's ~12 MB DLC is approximately 73% smaller than the ~45 MB baseline DLC. These latency ratios do not describe application throughput or camera FPS.

The CSV retains its original columns and adds configuration, evaluation dataset/count, Top-5 accuracy in percent, and calibration count. `accuracy_value` records Top-1 percent as named by `accuracy_metric`; `model_size_mb` contains approximate DLC sizes. Baseline calibration count is blank because no explicit quantization was performed. Runtime memory, throughput, and NPU utilization remain unmeasured.

## Key Lessons

- Both W8A8 and W8A16 preserved accuracy on this Imagenette validation set, with small observed differences. W8A8's positive difference does not establish that quantization inherently improves accuracy.
- W8A8 had the lowest measured latency: 0.310 ms versus 0.377 ms for W8A16 and 0.707 ms for the baseline, a 56.2% reduction and 2.28× inference speedup versus baseline.
- Quantized DLCs were approximately 12 MB versus approximately 45 MB for the baseline, largely attributable to reduced-precision weight storage. Artifact size does not establish runtime memory use.
- Increasing activation precision from INT8 to INT16 did not improve measured accuracy here. This does not imply that INT8 activations are universally more accurate than INT16.
- Small accuracy changes can result from numerical perturbations changing predictions near class decision boundaries; this is a possible explanation, not a measured per-image diagnosis.
- Evaluate precision choices empirically across accuracy, latency, model size, and runtime behavior. Higher numerical precision is not automatically preferable.

## Reproduction

Run from the repository root with the project environment, Qualcomm AI Hub credentials, model packages, and local datasets available:

```bash
python -m src.qualcomm.quantize_resnet18
python -m src.qualcomm.compile_resnet18
python -m src.qualcomm.compile_resnet18_w8a8
python -m src.qualcomm.compile_resnet18_w8a16
python -m src.validation.evaluate_imagenette
```

Quantization expects the original external-data package. Before quantized compilation/evaluation, make the downloaded QDQ graphs and any external tensors available in `resnet18_w8a8_package.onnx/` and `resnet18_w8a16_package.onnx/` under `models/resnet18/`; the evaluator expects `model.onnx` in each. Calibration images belong in `data/calibration/imagenette_samples/`, and validation images in `data/imagenette2/val/`. The calibration helper processes all matching images in its directory; use the separate 100-image calibration set for this experiment. Generated artifacts are not committed. Exact package versions and AI Hub job IDs are not recorded here.

## Future Model Ladder

ResNet-18 → MobileNet → object detection/YOLO → small Transformer → embedding model → small LLM → on-device RAG

The benchmark CSV uses task, input shape, precision, and named accuracy-metric fields to support future architectures. Leave uncollected measurements blank or `TBD` and document experiment-specific conditions in the notes.

## Experiment History

| Model | Experiment | Format | Estimated inference latency | Status |
| --- | --- | --- | --- | --- |
| ResNet-18 | baseline | QNN DLC | 0.707 ms | successful |
| ResNet-18 | W8A8 | QNN DLC | 0.310 ms | successful |
| ResNet-18 | W8A16 | QNN DLC | 0.377 ms | successful |
