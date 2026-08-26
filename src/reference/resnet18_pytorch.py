from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification


MODEL_ID = "microsoft/resnet-18"
IMAGE_PATH = "data/dog.jpg"

def main():
    
    #1. Load the image processor and model
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    
    print("")
    print("Model Class:")
    print(type(model))
        
    #2. Inference Mode
    model.eval()
    
    #3. Load and preprocess the image
    image = Image.open(IMAGE_PATH).convert("RGB")
    
    #4. Preprocess the image
    inputs = processor(images=image, return_tensors="pt")
    
    print("")
    print("Model Input:")
    print("pixel_values shape:", inputs["pixel_values"].shape)
    print("pixel_values dtype:", inputs["pixel_values"].dtype)
    print("")
    
    #5 Run the inference
    with torch.inference_mode():
        outputs = model(**inputs)
        
    logits = outputs.logits
    
    print("")
    print("Model Output:")
    print("logits shape:", logits.shape)
    print("logits dtype:", logits.dtype)
    print("")
    
    #6. Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=-1)
    
    #7. Get Top 5 predictions
    top5_probabilities, top5_indices = torch.topk(probabilities, k=5, dim=-1)
        
    print("\nTop 5 Predictions:")
    for prob, idx in zip(top5_probabilities[0], top5_indices[0]):
        label = model.config.id2label[idx.item()]
        print(f"\n{label:30s} {prob.item():.4%}")
        
if __name__ == "__main__":
    main()
        