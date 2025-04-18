# Import pipeline, torch, os, requests, and Image from PIL
from transformers import pipeline
import torch
import os
import requests
from PIL import Image

print("-------------------------------------------")
print("Hugging Face Local Inference Example")
print("Task: Image Captioning")
print("Model: nlpconnect/vit-gpt2-image-captioning")
print("-------------------------------------------")

# --- Image Handling ---
# 1. Define a path where the user *might* place their own image.
user_image_path = "image_captioning1.JPG" # <-- YOU CAN CHANGE THIS

# 2. Define a default image URL (surfers on beach) and download path.
#    Source: https://unsplash.com/photos/Tf-k_Dw19nU (by Austin Neill)
default_image_url = "https://hips.hearstapps.com/hmg-prod/images/180-random-facts-the-best-fun-facts-to-have-on-hand-67c1d5cba5e7c.jpg"
downloaded_image_path = "caption_sample_image.jpg"
image_to_process = None
headers = {'User-Agent': 'Mozilla/5.0'} # Some sites require a user-agent header

# 3. Decide which image to use
if os.path.exists(user_image_path):
    print(f"Using user-provided image: {user_image_path}")
    image_to_process = user_image_path
else:
    print(f"User image '{user_image_path}' not found.")
    print(f"Attempting to download sample image from Unsplash...")
    try:
        response = requests.get(default_image_url, headers=headers, stream=True, timeout=15)
        response.raise_for_status() # Raise an error for bad status codes
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
            if os.path.exists(downloaded_image_path): # Clean up corrupted download
                 os.remove(downloaded_image_path)

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Could not download sample image: {e}")
    except IOError as e:
        print(f"ERROR: Could not save downloaded image: {e}")

if image_to_process is None:
    print("\nERROR: No valid image available to process.")
    print(f"Please place an image at '{user_image_path}' or ensure internet connection to download sample.")
    exit()
# ------------------------

# --- Model Loading ---
print("\nLoading image captioning model (may download on first run)...")
try:
    # Use the "image-to-text" pipeline task
    captioner = pipeline(
        "image-to-text",
        model="nlpconnect/vit-gpt2-image-captioning", # Explicit model name
        device=0 if torch.cuda.is_available() else -1 # Use GPU if available, else CPU
        )
    print("Model loaded successfully.")
    if torch.cuda.is_available():
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure 'transformers', 'torch', 'Pillow', 'torchvision', 'timm' are installed.")
    exit()
# ----------------------

# --- Image Captioning ---
print(f"\nGenerating caption for '{os.path.basename(image_to_process)}'...")
try:
    # The pipeline handles loading, preprocessing, generation, and decoding
    # It returns a list of dictionaries, usually just one.
    captions = captioner(image_to_process)
    print("Caption generation complete.")

    # 5. Print the results
    print("\n--- Generated Caption(s) ---")
    if not captions:
        print("Could not generate caption for the image.")
    else:
        for i, caption_data in enumerate(captions):
             # Output structure is typically [{'generated_text': '...'}]
             caption_text = caption_data.get('generated_text', 'Caption format unexpected')
             print(f"Caption {i+1}: \"{caption_text}\"")
             print("-" * 20) # Separator

    print("-----------------------------")

except Exception as e:
    print(f"Error during image captioning: {e}")
    if isinstance(e, FileNotFoundError):
         print(f"Internal error: Could not find the image file at {image_to_process}")

# -----------------------

print("\nExample finished.")