from pathlib import Path

import torch
import onnx
from transformers import AutoModelForImageClassification


MODEL_ID = "microsoft/resnet-18"

OUTPUT_PATH = Path("models/resnet18/resnet18.onnx")


def main():

    # 1. Load the pretrained PyTorch model
    #
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)

    # Put the model into inference/evaluation mode.
    model.eval()

    # 2. Create the output folder
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


    # 3. Create an example input tensor
    dummy_input = torch.randn(1,3,224,224,dtype=torch.float32,)


    # 4. Export PyTorch → ONNX
    torch.onnx.export(
        model,                             # The PyTorch model we want to export.
        (dummy_input,),                       # Example input used to capture the computation.
        OUTPUT_PATH,                       # Destination path for the ONNX file.
        input_names=["pixel_values"],       # Give the ONNX input tensor a meaningful name.
        output_names=["logits"],            # Give the model output a meaningful name.
        opset_version=18,                   # ONNX operator-set version.Think of this as the version of the ONNX operator specification that the exported graph uses.
        training=torch.onnx.TrainingMode.EVAL, # We are exporting for inference only.
        do_constant_folding=True,           # Perform constant folding when possible. Example: if part of the graph can be precomputed because it never changes, the exporter may simplify it.
    )


    print(f"\nExported ONNX model to:")
    print(OUTPUT_PATH)


    # 5. Load the ONNX model back from disk
    onnx_model = onnx.load(OUTPUT_PATH)


    # 6. Validate the ONNX graph
    onnx.checker.check_model(onnx_model)

    print("\nONNX validation: PASSED")
    
    # 7. Inspect graph inputs
    print("\nGraph inputs:")

    for graph_input in onnx_model.graph.input:
        print(f"  {graph_input.name}")


    # 8. Inspect graph outputs

    print("\nGraph outputs:")
    for graph_output in onnx_model.graph.output:
        print(f"  {graph_output.name}")

    # 9. Count graph nodes
    print("\nNumber of graph nodes:")
    print(len(onnx_model.graph.node))

    # 10. Show unique operator types

    operator_types = sorted(set(node.op_type for node in onnx_model.graph.node))

    print("\nOperator types used:")

    for operator in operator_types:
        print(f"  {operator}")


if __name__ == "__main__":
    main()