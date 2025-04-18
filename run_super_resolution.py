# Import necessary libraries
from transformers import pipeline
import torch
import os
import requests
from PIL import Image

print("-------------------------------------------")
print("Hugging Face Local Inference Example")
print("Task: Image Super-Resolution (x2)")
print("Model: caidas/swin2SR-classical-sr-x2-64")
print("-------------------------------------------")

# --- USER ACTION RECOMMENDED ---
# 1. Define the path where your potentially lower-resolution image is.
#    For best effect, don't use an already massive image.
user_image_path = "low-res.jpg" # <-- CHANGE THIS

# -----------------------------

# --- Image Handling ---
# Use the same street scene sample image URL as before
default_image_url = "https://unsplash.com/photos/t倭YF547b0/download?force=true&w=640"
downloaded_image_path = "sr_sample_image.jpg"
image_to_process = None
headers = {'User-Agent': 'Mozilla/5.0'} # Header for requests

# Decide which image to use
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
        # Verify download
        try:
            img = Image.open(image_to_process)
            img.verify()
            print("Sample image verified.")
        except (IOError, SyntaxError) as e:
            print(f"Downloaded file is not valid: {e}")
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

# --- Load Input Image Info ---
try:
    input_img = Image.open(image_to_process)
    print(f"\nInput image: '{os.path.basename(image_to_process)}' (Dimensions: {input_img.width}x{input_img.height})")
    input_img.close() # Close file handle
except Exception as e:
    print(f"Error opening input image to get dimensions: {e}")
    exit()
# ---------------------------

# --- Model Loading ---
print("\nLoading Super-Resolution model (may download on first run)...")
try:
    # Use the "image-to-image" pipeline task
    upscaler = pipeline(
        "image-to-image", # Task for super-resolution, style transfer etc.
        model="caidas/swin2SR-classical-sr-x2-64", # Explicit Swin2SR model
        device=0 if torch.cuda.is_available() else -1
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

# --- Image Super-Resolution ---
print(f"\nPerforming 2x super-resolution...")
output_path = "super_resolution_output.png" # Save as PNG for lossless quality
try:
    # The pipeline takes the image path (or PIL image) and returns the upscaled PIL Image
    upscaled_image = upscaler(image_to_process)
    print("Super-resolution complete.")

    # 6. Save the resulting upscaled image
    if isinstance(upscaled_image, Image.Image):
        upscaled_image.save(output_path)
        print(f"\n--- Output ---")
        print(f"Upscaled image ({upscaled_image.width}x{upscaled_image.height}) saved successfully to: {output_path}")
        print("--------------")
    else:
        print("Could not process the image or unexpected output format from pipeline.")
        print(f"Pipeline output: {upscaled_image}") # Print output for debugging

except Exception as e:
    print(f"Error during super-resolution: {e}")
    if isinstance(e, FileNotFoundError):
         print(f"Internal error: Could not find the image file at {image_to_process}")
    print("Ensure image file is valid and check system/GPU memory.")

# --------------------------

print("\nExample finished.")