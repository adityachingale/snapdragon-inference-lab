from pathlib import Path
import qai_hub as hub

# ================================================
# Configuration
# ================================================
ONNX_PACKAGE_PATH = Path("models/resnet18/resnet18_w8a8_package.onnx")
DEVICE_NAME = "Samsung Galaxy S24 (Family)"
OUTPUT_PATH = Path("models/resnet18/resnet18_w8a8_qnn.dlc")

def main():
    
    if not ONNX_PACKAGE_PATH.is_dir():
        raise FileNotFoundError(f"W8A8 ONNX package not found: {ONNX_PACKAGE_PATH}")
    
    # Create Qualcomm AI Hub client
    client = hub.Client()
    
    # Define Target Device
    device = hub.Device(DEVICE_NAME)
    
    print(f"Compiling {ONNX_PACKAGE_PATH} for {DEVICE_NAME}...")
    
    # Submit a compilation job to Qualcomm AI Hub
    compile_job = client.submit_compile_job(
                model=str(ONNX_PACKAGE_PATH),
                device=device,
                name="resnet18_w8a8_qnn_s24",
                options="--target_runtime qnn_dlc",
            )
    
    print("\nCompile job submitted.")
    print(compile_job)
    
    # Wait for the compilation job to complete
    status = compile_job.wait()
    if not status.success:
        raise RuntimeError(f"Compilation failed: {status}. Job: {compile_job.url}")
    
    target_model = compile_job.get_target_model()
    if target_model is None:
        raise RuntimeError(f"Compilation produced no target model. Job: {compile_job.url}")
    
    print("Compilation completed. Downloading the compiled model...")
    print("Target Model: ", target_model)
    
    # Download the compiled model
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    target_model.download(str(OUTPUT_PATH))
    
    print(f"Compiled model downloaded to {OUTPUT_PATH}")
    
    # ============================================================
    # Profile the compiled QNN model
    # ============================================================

    print("\nSubmitting profiling job...")

    profile_job = client.submit_profile_job(
                        model=target_model,
                        device=device,
                        name="resnet18_w8a8_qnn_s24_profile",
                    )

    print(f"Profile job submitted: {profile_job.url}")

    # Wait for profiling to finish
    profile_status = profile_job.wait()

    print(f"\nProfile job status: {profile_status}")

    if not profile_status.success:
        raise RuntimeError(
            f"Profiling failed: {profile_status}. "
            f"Job: {profile_job.url}"
        )

    # Download profiling results
    profile = profile_job.download_profile()

    print("\n========== PROFILE RESULTS ==========")
    print(profile)
    
    latency_us = profile["execution_summary"]["estimated_inference_time"]
    latency_ms = latency_us / 1000

    print(f"\nEstimated inference latency: {latency_ms:.3f} ms")
    
    
if __name__ == "__main__":
    main()
