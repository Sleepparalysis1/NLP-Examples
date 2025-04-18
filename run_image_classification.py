# Import the pipeline function, torch, and os for path checking
from transformers import pipeline
import torch
import os

print("-------------------------------------------")
print("Hugging Face Local Inference Example")
print("Task: Image Classification")
print("Model: google/vit-base-patch16-224 (Vision Transformer)")
print("-------------------------------------------")

# --- IMPORTANT: USER ACTION REQUIRED ---
# 1. Define the path to YOUR local image file.
#    Replace the placeholder below with the actual path to an image
#    on your computer (e.g., a .jpg, .png file).
#    Examples:
#    image_path = "/home/your_user/Pictures/cat.jpg"
#    image_path = "C:/Users/YourUser/Pictures/dog.png"
#    image_path = "./my_image.jpg" # If image is in the same folder as the script

image_path = "./image.jpeg" # <-- CHANGE THIS LINE

# -----------------------------------------

# 2. Check if the image file exists before proceeding
if not os.path.exists(image_path):
    print(f"\nERROR: Image file not found at path: {image_path}")
    print("Please update the 'image_path' variable in the script with a valid path.")
    exit()
else:
    print(f"\nUsing image file: {image_path}")


# 3. Load the image classification pipeline, explicitly specifying the ViT model
print("Loading image classification model (may download on first run)...")
try:
    # Use the "image-classification" pipeline task
    image_classifier = pipeline(
        "image-classification", # Standard task name
        model="google/vit-base-patch16-224", # Explicit model name
        device=0 if torch.cuda.is_available() else -1 # Use GPU if available, else CPU
        )
    print("Model loaded successfully.")
    if torch.cuda.is_available():
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU.")

except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'transformers', 'torch', 'Pillow', 'torchvision', and 'timm' are installed.")
    exit()


# 4. Run the image classification pipeline
#    We can ask for the top N predictions using top_k
num_predictions = 5
print(f"\nClassifying image (Top {num_predictions} predictions)...")
try:
    # The pipeline handles loading, preprocessing, inference, and postprocessing
    predictions = image_classifier(image_path, top_k=num_predictions)
    print("Classification complete.")

    # 5. Print the results
    #    The result is a list of dictionaries, each containing 'label' and 'score'
    print(f"\n--- Top {num_predictions} Predictions for '{os.path.basename(image_path)}' ---")
    if not predictions:
        print("Could not classify the image.")
    else:
        for i, prediction in enumerate(predictions):
            label = prediction['label'] # Predicted class label (from ImageNet)
            score = prediction['score'] # Confidence score

            print(f"{i+1}. Label: {label}")
            print(f"   Confidence: {score:.4f}")
            print("-" * 15) # Separator

    print("----------------------------------------------")

except Exception as e:
    print(f"Error during image classification: {e}")


print("\nExample finished.")