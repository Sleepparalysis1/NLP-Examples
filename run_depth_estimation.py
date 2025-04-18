# Import necessary libraries
from transformers import pipeline
import torch
import os
import requests
from PIL import Image

print("-------------------------------------------")
print("Hugging Face Local Inference Example")
print("Task: Depth Estimation")
print("Model: Intel/dpt-large (Dense Prediction Transformer)")
print("-------------------------------------------")

# --- Image Handling ---
# 1. Define a path where the user *might* place their own image.
user_image_path = "my_segmentation_image.jpg" # <-- YOU CAN CHANGE THIS

# 2. Define a default image URL (city street view) and download path.
#    Source: https://unsplash.com/photos/t倭YF547b0 (by Charles POSTIAUX)
default_image_url = "https://images.unsplash.com/photo-1744882838449-b3ad2ceff9a8?q=80&w=1974"
downloaded_image_path = "depth_sample_image.jpg"
image_to_process = None
headers = {'User-Agent': 'Mozilla/5.0'} # Header for requests

# 3. Decide which image to use
if os.path.exists(user_image_path):
    print(f"Using user-provided image: {user_image_path}")
    image_to_process = user_image_path
else:
    print(f"User image '{user_image_path}' not found.")
    print(f"Attempting to download sample image from Unsplash...")
    try:
        response = requests.get(default_image_url, headers=headers, stream=True, timeout=15)
        response.raise_for_status()
        with open(downloaded_image_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Sample image downloaded successfully as {downloaded_image_path}")
        image_to_process = downloaded_image_path
        # Verify download is a valid image
        try:
            img = Image.open(image_to_process)
            img.verify()
            print("Sample image verified.")
        except (IOError, SyntaxError) as e:
            print(f"Downloaded file is not a valid image or is corrupted: {e}")
            image_to_process = None
            if os.path.exists(downloaded_image_path): os.remove(downloaded_image_path)
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Could not download sample image: {e}")
    except IOError as e:
        print(f"ERROR: Could not save downloaded image: {e}")

if image_to_process is None:
    print("\nERROR: No valid image available to process.")
    exit()
# ------------------------

# --- Model Loading ---
print("\nLoading Depth Estimation model (may download on first run)...")
try:
    # Use the "depth-estimation" pipeline task
    depth_estimator = pipeline(
        "depth-estimation",
        model="Intel/dpt-large", # Explicit DPT model name
        device=0 if torch.cuda.is_available() else -1 # Use GPU if available, else CPU
        )
    print("Model loaded successfully.")
    if torch.cuda.is_available():
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure relevant libraries are installed: transformers, torch, Pillow, torchvision, timm...")
    exit()
# ----------------------

# --- Depth Estimation ---
print(f"\nEstimating depth for '{os.path.basename(image_to_process)}'...")
output_path = "depth_map_output.png" # Save as PNG
try:
    # The pipeline returns a dictionary, usually with 'predicted_depth' (Tensor)
    # and 'depth' (PIL.Image representation)
    result = depth_estimator(image_to_process)
    print("Depth estimation complete.")

    # 6. Save the resulting depth map image
    if result and 'depth' in result and isinstance(result['depth'], Image.Image):
        depth_map_image = result['depth']
        depth_map_image.save(output_path)
        print(f"\n--- Output ---")
        print(f"Predicted depth map saved successfully to: {output_path}")
        print(f"(Image size: {depth_map_image.size[0]}x{depth_map_image.size[1]})")
        print("Note: In the output PNG, pixel intensity usually relates to depth.")
        print("----------------")
    else:
        print("Could not extract depth map image from pipeline result.")
        print(f"Pipeline output: {result}") # Print full output for debugging

except Exception as e:
    print(f"Error during depth estimation: {e}")
    if isinstance(e, FileNotFoundError):
         print(f"Internal error: Could not find the image file at {image_to_process}")

# -----------------------

print("\nExample finished.")