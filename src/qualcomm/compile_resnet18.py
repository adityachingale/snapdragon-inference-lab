from pathlib import Path
import qai_hub as hub
import shutil

# ================================================
# Configuration
# ================================================
ONNX_PATH = Path("models/resnet18/resnet18.onnx")
DEVICE_NAME = "Samsung Galaxy S24 (Family)" # Qualcomm AI Hub's Device name for the target device
OUTPUT_PATH = Path("models/resnet18/resnet18_qnn.dlc") # Output path for the compiled model
ONNX_PACKAGE_PATH = Path("models/resnet18/resnet18_package.onnx")

def main():
    
    # Our ONNX model stores its weights in a separate .data file.
    external_data_path = Path(f"{ONNX_PATH}.data")

    # Make sure both parts of the ONNX model exist.
    if not ONNX_PATH.is_file():
        raise FileNotFoundError(f"ONNX model not found: {ONNX_PATH}")

    if not external_data_path.is_file():
        raise FileNotFoundError(
            f"External ONNX weight data not found: {external_data_path}"
        )

    # Qualcomm expects an external-data ONNX model to be uploaded
    # as a directory ending in ".onnx".
    #
    # The directory will contain:
    #
    # resnet18_package.onnx/
    #     resnet18.onnx
    #     resnet18.onnx.data
    #
    if ONNX_PACKAGE_PATH.exists():
        shutil.rmtree(ONNX_PACKAGE_PATH)

    ONNX_PACKAGE_PATH.mkdir(parents=True)

    shutil.copy2(
        ONNX_PATH,
        ONNX_PACKAGE_PATH / ONNX_PATH.name,
    )

    shutil.copy2(
        external_data_path,
        ONNX_PACKAGE_PATH / external_data_path.name,
    )

    print(f"Qualcomm ONNX package created at: {ONNX_PACKAGE_PATH}")
    
    # Create Qualcomm AI Hub client
    client = hub.Client()
    
    # Define Target Device
    device = hub.Device(DEVICE_NAME)
    
    print(f"Compiling {ONNX_PACKAGE_PATH} for {DEVICE_NAME}...")
    
    # Submit a compilation job to Qualcomm AI Hub
    compile_job = client.submit_compile_job(
                model=str(ONNX_PACKAGE_PATH),
                device=device,
                name="resnet18_qnn_s24",
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
    
if __name__ == "__main__":
    main()
