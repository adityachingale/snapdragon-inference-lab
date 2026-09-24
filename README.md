# snapdragon-inference-lab

Experiments in deploying and optimizing pretrained ML and LLM models on Qualcomm Snapdragon NPUs. The completed ResNet-18 milestone covers Hugging Face/PyTorch → ONNX → Qualcomm compilation and quantization → Android deployment → correctness validation and profiling on a physical Samsung Galaxy S24.

The project demonstrates model-conversion validation, calibration versus evaluation, QNN runtime integration, tensor-layout debugging, and performance analysis across runtime and accelerator measurement boundaries. See the [ResNet-18 experiment log](results/notes/resnet18.md) for detailed results and limitations.

## Hardware and software

| Component | Experiment environment |
| --- | --- |
| Host | Apple Silicon Mac |
| Host conversion tools | Linux x86-64 QAIRT in an amd64 Docker environment |
| Physical target | Samsung Galaxy S24; Snapdragon 8 Gen 3 / SM8650 |
| Android application architecture | arm64-v8a |
| Accelerator | Hexagon v75 HTP/NPU |
| Physical deployment SDK | QAIRT 2.50.0.260828 |
| NetRun, profile viewer, backend | v2.50.0.260828221209 |

The Mac, Linux host tools, Android ARM64 runtime, and Hexagon accelerator are distinct execution environments. [docker/qairt/Dockerfile](docker/qairt/Dockerfile) provides the amd64 host-tool environment and Android NDK r26c; it does not bundle the QAIRT SDK.

## Deployment paths

```text
Hugging Face / PyTorch → ONNX opset 18 + external weights
  ├─ Qualcomm AI Hub → floating / W8A16 / W8A8 QDQ → QNN DLC
  └─ Local QAIRT converter → .cpp + .bin + metadata → Android libresnet18.so
       ↓ both paths
Android ARM64 qnn-net-run + model loader → QNN HTP backend
       ↓ FastRPC
Hexagon v75 HTP on the physical Galaxy S24
```

QDQ applies to the quantized variants. A DLC is a Qualcomm model representation consumed by QNN, not simply an NPU executable. The two compilation paths are evaluated separately.

## Repository structure

| Path | Purpose |
| --- | --- |
| [src/reference/](src/reference/) | PyTorch reference inference |
| [src/export/](src/export/) | ONNX export |
| [src/data/](src/data/) | Calibration preprocessing |
| [src/qualcomm/](src/qualcomm/) | AI Hub quantization/compilation/profiling and local QNN input preparation |
| [src/validation/](src/validation/) | Conversion checks, Imagenette evaluation, physical-output comparison |
| [src/benchmarking/analyze_qnn_profiles.py](src/benchmarking/analyze_qnn_profiles.py) | Physical QNN profile analysis |
| [docker/](docker/) | Linux host-tool environments |
| [results/notes/resnet18.md](results/notes/resnet18.md) | Detailed experiment history and physical benchmark results |
| [results/benchmark_results.csv](results/benchmark_results.csv) | AI Hub reference latency and Imagenette results; not physical latency |

## ResNet-18 validation and quantization

`microsoft/resnet-18` consumes FP32 NCHW `[1, 3, 224, 224]` and produces 1,000 ImageNet logits. Export uses ONNX opset 18; both `resnet18.onnx` and `resnet18.onnx.data` are required for external-data packaging.

PyTorch versus ONNX on the reference image: max absolute difference **1.14441e-5**, mean **1.7367e-6**, RMSE **2.2637e-6**, with Top-1 agreement. This established numerical equivalence for that input before Qualcomm deployment; it is separate from dataset-level accuracy.

W8A16 requests INT8 weights and INT16 activations; W8A8 requests INT8 weights and INT8 activations. Each used 100 representative calibration images, separate from evaluation. Floating denotes a pre-explicit-quantization baseline, not verified FP32 execution throughout QNN.

| AI Hub variant | Imagenette Top-1 | Imagenette Top-5 | Approximate DLC size | AI Hub estimated latency |
| --- | --- | --- | --- | --- |
| Floating | 81.45% | 95.97% | ~45 MB | ~0.707 ms |
| W8A16 | 81.48% | 95.92% | ~12 MB | ~0.377 ms |
| W8A8 | 81.89% | 96.10% | ~12 MB | ~0.310 ms |

Accuracy was measured with ONNX Runtime CPU on 3,925 full-size Imagenette validation images, using the same slow Hugging Face preprocessing. Quantization preserved accuracy on this set; small fluctuations are not evidence of improved model accuracy. These are not full ImageNet validation results. Smaller artifacts alone do not explain all latency changes.

## Physical deployment and correctness

Both the local QAIRT model library and AI Hub DLCs executed through QNN HTP on the physical S24. Android-side dependencies included `libQnnHtp.so`, `libQnnHtpV75Stub.so`, and `libQnnHtpPrepare.so`, communicating through FastRPC with the v75 skeleton/runtime. A missing `libQnnHtpPrepare.so` allowed initialization to progress but blocked graph creation/preparation.

**Tensor layout was part of the deployment contract:** the local `.so` exposed NHWC `[1, 224, 224, 3]`, while these AI Hub DLCs accepted the original NCHW input. Feeding NCHW bytes to the local NHWC interface executed without necessarily raising an error, but produced incorrect classifications. Transposing corrected the output. Raw tensors carry no layout metadata; these observations apply to the specific artifacts, not all QAIRT or AI Hub models.

PyTorch, ONNX, and the corrected local QNN run agreed on Top-1 and Top-5 class ordering. Physical W8A16 preserved the reference Top-5 order; W8A8 preserved Top-1 and the first four ranks, with a low-probability fifth-class change. Higher confidence is not higher accuracy. [Detailed correctness comparisons](results/notes/resnet18.md#physical-correctness-validation) retain the numerical results.

## Physical benchmark

Each profile used QNN HTP with `--perf_profile=high_performance` and `--profiling_level=basic`: 100 inference executions, first five excluded, 95 steady-state observations per variant.

**NetRun** is the broader runtime execution measurement. **Accelerator** below means the narrower `Accelerator (execute excluding wait)` event. Neither is complete application latency or an unqualified “pure model latency”; no application FPS is inferred.

| Model | NetRun mean | NetRun P95 | Accelerator mean | Accelerator P95 |
| --- | --- | --- | --- | --- |
| Local QAIRT FP | 5.914 ms | 6.042 ms | 2.442 ms | 2.466 ms |
| AI Hub FP | 6.882 ms | 7.163 ms | 3.489 ms | 3.702 ms |
| W8A16 | 5.481 ms | 5.739 ms | 1.966 ms | 2.169 ms |
| W8A8 | 4.872 ms | 5.219 ms | 1.429 ms | 1.668 ms |

Use **AI Hub FP as the quantization baseline**, because all three variants share the AI Hub/DLC path. W8A16 achieved **1.77× accelerator** and **1.26× NetRun** speedups. W8A8 reduced mean accelerator latency from 3.489 to 1.429 ms (**59.04% / 2.44×**) and NetRun from 6.882 to 4.872 ms (**29.20% / 1.41×**).

The smaller NetRun gains demonstrate the effect of remaining runtime overhead: RPC, synchronization, waits, and tensor handling are not proportionally accelerated. This experiment does not attribute a measured share to each component.

Local QAIRT FP was faster than AI Hub FP in this run, but the different compilation paths prevent attributing that result to `.so` versus DLC alone. AI Hub estimates and physical measurements also have different boundaries/configurations; compare their consistent ordering (**W8A8 < W8A16 < Floating**), not their absolute numbers as equivalent measurements.

## Reproducing the checks

From the repository root, using a Python environment with the scripts' dependencies and the required local model/data artifacts:

```bash
python -m src.validation.compare_pytorch_onnx
python -m src.qualcomm.prepare_qnn_input
python -m src.validation.compare_pytorch_onnx_qnn
python -m src.validation.evaluate_imagenette
python -m src.benchmarking.analyze_qnn_profiles
```

Input preparation writes NHWC for the local model library; it is not an NCHW generator for the DLCs. Three-way validation requires the pulled local QNN `logits.raw`. The analyzer requires four profiler CSVs under `data/qnn_profiles/` and reports mean, median, P90/P95/P99, and other statistics. See [reproduction details and gaps](results/notes/resnet18.md#reproduction) for inputs, AI Hub commands, and prerequisites. Device push/run/pull and local conversion commands are not yet captured as repository scripts.

## Status and next steps

Completed: ResNet-18 export, conversion validation, W8A8/W8A16 calibration and Imagenette evaluation, two Qualcomm deployment paths, physical correctness checks, and physical profiling.

Planned model ladder: ResNet-18 → MobileNet → object detection/YOLO → small Transformer → embedding model → small LLM → on-device RAG. Remaining reproducibility work includes environment pins, automated device setup, artifact provenance, and controlled repeated profiling. [Publication gaps](results/notes/resnet18.md#reproducibility-and-publication-gaps) include generated artifacts already tracked despite ignore rules.
