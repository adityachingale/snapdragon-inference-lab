# ResNet-18 — Snapdragon Benchmark

## Objective

Establish a baseline for progressively testing model architectures and quantization strategies on Qualcomm Snapdragon hardware. Results below record the completed experiment; this documentation update did not rerun device inference, evaluation, compilation, or profile collection. The saved physical profiler CSVs were reanalyzed with the repository analyzer. Measurements and runtime observations are from the completed experiments reported by the project author.

## Model

- Model ID: `microsoft/resnet-18`
- Task: ImageNet-1K classification
- Input: `1x3x224x224`, float32, NCHW
- Output: `1x1000` logits

## PyTorch Reference

[`src/reference/resnet18_pytorch.py`](../../src/reference/resnet18_pytorch.py) loads the Hugging Face image processor and pretrained model, preprocesses `data/dog.jpg` as RGB, and runs PyTorch inference in evaluation mode with gradients disabled. Softmax probabilities are used to display the top-5 predictions.

## ONNX Export

[`src/export/resnet18_to_onnx.py`](../../src/export/resnet18_to_onnx.py) exports the PyTorch model to ONNX with a float32 example input and opset 18, then checks the ONNX graph. The exporter produced external tensor storage:

- `resnet18.onnx`: model graph
- `resnet18.onnx.data`: external tensor weights

Ignore rules cover these generated artifacts, but some are already tracked; see the publication gaps below.

## ONNX Validation

[`src/validation/compare_pytorch_onnx.py`](../../src/validation/compare_pytorch_onnx.py) compares PyTorch and ONNX Runtime CPU outputs using the same preprocessed test image.

| Check | Result |
| --- | --- |
| Input shape | `[1, 3, 224, 224]` |
| Output shape | `[1, 1000]` |
| Max absolute difference | 0.0000114441 |
| Mean absolute difference | 0.0000017367 |
| RMSE | 0.0000022637 |
| Top-1 agreement | true |
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

## AI Hub Baseline Profiling

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

## AI Hub Reference Benchmark Summary

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
- In AI Hub reference profiling, W8A8 had the lowest estimated latency: 0.310 ms versus 0.377 ms for W8A16 and 0.707 ms for the baseline, a 56.2% reduction and 2.28× inference speedup versus baseline.
- Quantized DLCs were approximately 12 MB versus approximately 45 MB for the baseline, largely attributable to reduced-precision weight storage. Artifact size does not establish runtime memory use.
- Increasing activation precision from INT8 to INT16 did not improve measured accuracy here. This does not imply that INT8 activations are universally more accurate than INT16.
- Small accuracy changes can result from numerical perturbations changing predictions near class decision boundaries; this is a possible explanation, not a measured per-image diagnosis.
- Evaluate precision choices empirically across accuracy, latency, model size, and runtime behavior. Higher numerical precision is not automatically preferable.

## Physical Deployment Environment

| Component | Value |
| --- | --- |
| Physical device | Samsung Galaxy S24 |
| SoC / accelerator | Snapdragon 8 Gen 3 / SM8650; Hexagon v75 HTP/NPU |
| Android architecture | arm64-v8a |
| Host | Apple Silicon Mac |
| Host tools | Linux x86-64 QAIRT through amd64 Docker |
| Physical-runtime QAIRT SDK | 2.50.0.260828 |
| qnn-net-run / qnn-profile-viewer / backend | v2.50.0.260828221209 |

The physical handset is distinct from the AI Hub “Samsung Galaxy S24 (Family)” target. The Mac host, amd64 conversion tools, Android ARM64 application runtime, and Hexagon accelerator have different architectures. [`docker/qairt/Dockerfile`](../../docker/qairt/Dockerfile) sets up the amd64 environment and Linux-host Android NDK r26c; the QAIRT SDK must be supplied separately. [`docker/Dockerfile.qnn`](../../docker/Dockerfile.qnn) is a more minimal Ubuntu environment.

### Execution stack and missing dependency

Android ARM64 runs `qnn-net-run`, the model/DLC loader, `libQnnHtp.so`, `libQnnHtpV75Stub.so`, `libQnnHtpPrepare.so`, and supporting QNN libraries. FastRPC connects this runtime to the Hexagon v75 HTP/DSP skeleton and runtime libraries. A DLC is a Qualcomm model representation consumed by QNN, rather than simply an “NPU executable.”

During debugging, `libQnnHtpPrepare.so` was a critical missing dependency: device/context/FastRPC initialization could progress, but graph creation/preparation failed. Restoring the preparation library was necessary for this deployment.

### Separate local QAIRT path

The local path was ONNX → QNN converter → generated `.cpp`, `.bin`, and model metadata → Android model-library generation → `libresnet18.so` → `qnn-net-run` → HTP backend → physical S24. Generated files are present under `models/resnet18/qairt_local/`, including `model_libs/aarch64-android/libresnet18.so`. This model successfully executed on the Snapdragon 8 Gen 3 / Hexagon v75 device. This path is separate from the AI Hub floating and quantized DLCs.

### Tensor-layout debugging

| Artifact | Application-facing input observed in this experiment |
| --- | --- |
| Hugging Face / source ONNX | FP32 NCHW `[1, 3, 224, 224]` |
| Local QAIRT model library | FP32 NHWC `[1, 224, 224, 3]` |
| AI Hub DLC variants | Original FP32 NCHW input |

Generated local metadata recorded a permutation back toward the source layout. NCHW and NHWC tensors have the same element count, so passing the original NCHW bytes to the local NHWC interface did not necessarily cause a runtime error: execution succeeded with incorrect classifications. Transposing NCHW → NHWC corrected physical-device output.

[`prepare_qnn_input.py`](../../src/qualcomm/prepare_qnn_input.py) applies the reference processor, transposes with `(0, 2, 3, 1)`, and writes contiguous FP32 NHWC bytes. The AI Hub DLCs were validated using the original NCHW raw input instead. These are observations for these artifacts, not universal layout rules for QAIRT libraries or AI Hub DLCs.

**Successful execution does not prove inference correctness.** Raw tensor files carry no layout metadata. Verify the deployed artifact's application-facing tensor interface; matching byte count alone cannot establish correct semantics.

## Physical Correctness Validation

[`compare_pytorch_onnx_qnn.py`](../../src/validation/compare_pytorch_onnx_qnn.py) compares the reference image's logits with the pulled local QNN output. [`decode_qnn_output.py`](../../src/validation/decode_qnn_output.py) decodes the local output's Top-5.

| Comparison | Max absolute difference | Mean absolute difference | RMSE | Top-1 agreement |
| --- | --- | --- | --- | --- |
| PyTorch vs ONNX | 0.0000114441 | 0.0000017367 | 0.0000022637 | true |
| PyTorch vs physical local QNN | 0.0219650269 | 0.0026150353 | 0.0034260151 | true |
| ONNX vs physical local QNN | 0.0219707489 | 0.0026156635 | 0.0034269283 | true |

All three produced the same Top-5 classes and order for the reference image after correcting the local layout. This is a single-image correctness check, not dataset-level physical-device accuracy.

Physical AI Hub DLC checks produced the following softmax probabilities:

| Rank / class | PyTorch | AI Hub floating DLC | W8A16 DLC | W8A8 DLC |
| --- | --- | --- | --- | --- |
| 1. golden retriever | 96.9484% | 96.9099% | 96.9256% | 97.2642% |
| 2. Great Pyrenees | 0.6434% | 0.6479% | 0.6325% | 0.5691% |
| 3. Labrador retriever | 0.4492% | 0.4523% | 0.4606% | 0.4243% |
| 4. Tibetan terrier / chrysanthemum dog | 0.4236% | 0.4316% | 0.4128% | 0.3663% |
| 5. Brittany spaniel (except W8A8) | 0.2096% | 0.2120% | 0.2218% | kuvasz: 0.1757% |

Floating and W8A16 preserved the full reference Top-5 order. W8A8 preserved Top-1 and the first four ranks; only the very low-probability fifth class changed. Its higher golden-retriever confidence does not imply higher accuracy.

## Physical Benchmark Methodology

The physical Samsung Galaxy S24 used QNN HTP, `--perf_profile=high_performance`, and `--profiling_level=basic`. All four variants used the same methodology: 100 executions per profile, first five excluded, leaving 95 observations for steady-state statistics. Initialization, graph preparation, and finalization are not included in these execution-event distributions.

[`src/benchmarking/analyze_qnn_profiles.py`](../../src/benchmarking/analyze_qnn_profiles.py) reads the four `profile.csv` files under `data/qnn_profiles/`, validates 100 events per boundary, converts microseconds to milliseconds, removes five warm-up observations, and reports mean, median, P90/P95/P99, min/max, and standard deviation.

- **NetRun:** `EXECUTE`, timing source `NETRUN`, event starting with `Graph `; a broader runtime execution measurement.
- **Accelerator:** `EXECUTE`, source `BACKEND`, event `Accelerator (execute excluding wait) time`; a narrower accelerator-compute-oriented measurement.

Neither boundary is unqualified “pure model latency” or a complete application/camera pipeline. No FPS is inferred. AI Hub reference estimates use different boundaries/configurations and must remain separate from these measurements.

### Steady-state results

| Model | NetRun mean | NetRun P95 | Accelerator mean | Accelerator P95 |
| --- | --- | --- | --- | --- |
| Local QAIRT FP | 5.914 ms | 6.042 ms | 2.442 ms | 2.466 ms |
| AI Hub FP | 6.882 ms | 7.163 ms | 3.489 ms | 3.702 ms |
| W8A16 | 5.481 ms | 5.739 ms | 1.966 ms | 2.169 ms |
| W8A8 | 4.872 ms | 5.219 ms | 1.429 ms | 1.668 ms |

**AI Hub FP is the primary quantization baseline**: it shares the AI Hub/DLC deployment path with W8A16 and W8A8. Reductions and speedups below use unrounded steady-state means, not the rounded table entries.

| Comparison | NetRun reduction | NetRun speedup | Accelerator reduction | Accelerator speedup |
| --- | --- | --- | --- | --- |
| W8A16 vs AI Hub FP | 20.36% | 1.26× | 43.65% | 1.77× |
| W8A8 vs AI Hub FP | 29.20% | 1.41× | 59.04% | 2.44× |
| W8A8 vs W8A16 | 11.10% | 1.12× | 27.32% | 1.38× |

The ordering W8A8 < W8A16 < floating holds for both physical boundaries and for AI Hub reference estimates, despite different absolute latencies. Quantization preserved Imagenette accuracy and substantially reduced artifact size, but smaller size alone does not establish the cause of all speedups.

The accelerator gains exceed NetRun gains because RPC, synchronization, runtime work, waits, tensor handling, and other overhead remain. These profiles demonstrate the effect of remaining overhead; they do not isolate each component's contribution or measure whole-application speedup.

### Deployment-path comparison

Local QAIRT FP measured 5.914 ms NetRun and 2.442 ms accelerator means, versus 6.882 ms and 3.489 ms for AI Hub FP: **14.07%** and **30.03%** lower, respectively. This is a separate observed deployment-path difference, not a claim that `.so` is inherently faster than DLC. Compiler configuration, graph transformations, internal precision, representation, and runtime preparation could contribute. Controlled experiments are required to isolate the cause.

## Reproduction

Run from the repository root with the project environment, Qualcomm AI Hub credentials, model packages, and local datasets available:

```bash
python -m src.qualcomm.quantize_resnet18
python -m src.qualcomm.compile_resnet18
python -m src.qualcomm.compile_resnet18_w8a8
python -m src.qualcomm.compile_resnet18_w8a16
python -m src.validation.evaluate_imagenette
```

Quantization expects the original external-data package. Before quantized compilation/evaluation, make the downloaded QDQ graphs and any external tensors available in `resnet18_w8a8_package.onnx/` and `resnet18_w8a16_package.onnx/` under `models/resnet18/`; the evaluator expects `model.onnx` in each. Calibration images belong in `data/calibration/imagenette_samples/`, and validation images in `data/imagenette2/val/`. The calibration helper processes all matching images in its directory; use the separate 100-image calibration set for this experiment. Generated artifacts should remain outside version control; some previously tracked files still need review. Exact package versions and AI Hub job IDs are not recorded here.

### Physical validation and profile analysis

With reference artifacts and dependencies available, run from the repository root:

```bash
python -m src.validation.compare_pytorch_onnx
python -m src.qualcomm.prepare_qnn_input
python -m src.validation.compare_pytorch_onnx_qnn
python -m src.validation.decode_qnn_output
python -m src.benchmarking.analyze_qnn_profiles
```

The input helper writes `data/qnn_inputs/dog_pixel_values_nhwc.raw` for the local `.so` only. Both physical-output scripts currently read `data/qnn_outputs/resnet18_local/output/Result_0/logits.raw`; neither is a CLI for selecting all DLC outputs. Pull the local device output there before comparing. The analyzer expects `profile.csv` in each of `resnet18_fp32`, `resnet18_aihub_fp`, `resnet18_w8a16`, and `resnet18_w8a8` under `data/qnn_profiles/`. The `resnet18_fp32` directory name does not establish internal FP32 execution.

### Reproducibility and publication gaps

- Exact converter, model-library generation, device push/run/pull, and profile-viewer invocations are not captured in reusable repository scripts. Only verified script commands and reported profiling flags are documented here.
- A dedicated NCHW raw-input generator for the AI Hub DLCs is missing; the existing helper writes NHWC. SDK provisioning, library search paths, and the complete target library manifest still need a reproducible deployment recipe.
- Python dependency pins are incomplete (`pyproject.toml` is empty); AI Hub job IDs, artifact hashes, and converter options should be recorded. The physical QAIRT/runtime versions above are recorded, and profile headers confirm runtime versions.
- Device thermal state, power conditions, run ordering, and repeated independent sessions are not recorded here. The 95 observations describe each saved profile, not cross-session reproducibility or statistical significance.
- `.gitignore` covers profile directories, raw inputs/outputs, DLCs, shared libraries, weights, and local QAIRT build output. However, Git already tracks some ONNX weights, raw tensors/output metadata, and generated local `.cpp`/JSON files; ignore rules do not untrack them. Review before publication. This documentation-only update does not remove or stage artifacts. Markdown and benchmark-summary CSVs remain eligible for version control.
- The benchmark analyzer and three-way comparison script were present but untracked during this update; include the intended source files when publishing. Raw profiler CSVs are ignored, so public readers need regenerated profiles to rerun the analyzer. The existing benchmark CSV remains an AI Hub reference summary; physical results are recorded in this log.

## Future Model Ladder

ResNet-18 → MobileNet → object detection/YOLO → small Transformer → embedding model → small LLM → on-device RAG

The benchmark CSV uses task, input shape, precision, and named accuracy-metric fields to support future architectures. Leave uncollected measurements blank or `TBD` and document experiment-specific conditions in the notes.

## AI Hub Experiment History

| Model | Experiment | Format | Estimated inference latency | Status |
| --- | --- | --- | --- | --- |
| ResNet-18 | baseline | QNN DLC | 0.707 ms | successful |
| ResNet-18 | W8A8 | QNN DLC | 0.310 ms | successful |
| ResNet-18 | W8A16 | QNN DLC | 0.377 ms | successful |
