# Import necessary libraries
from transformers import pipeline
import torch
import os
import requests
from PIL import Image

# Check for optional text processing libraries often used by CLIP
try:
    import ftfy
    print("ftfy library found.")
except ImportError:
    print("Warning: ftfy library not found. Install it: pip install ftfy")
    print("CLIP text processing might be affected.")
try:
    import regex
    print("regex library found.")
except ImportError:
     print("Warning: regex library not found. Install it: pip install regex")
     print("CLIP text processing might be affected.")


print("-------------------------------------------")
print("Hugging Face Local Inference Example")
print("Task: Zero-Shot Image Classification")
print("Model: openai/clip-vit-base-patch32 (CLIP)")
print("-------------------------------------------")

# --- Image Handling ---
# 1. Define a path where the user *might* place their own image.
user_image_path = "my_zero_shot_image.jpg" # <-- YOU CAN CHANGE THIS

# 2. Define the same default image URL (COCO cats/remote) and download path.
default_image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
downloaded_image_path = "zero_shot_sample_image.jpg"
image_to_process = None
headers = {'User-Agent': 'Mozilla/5.0'} # Header for requests

# 3. Decide which image to use
if os.path.exists(user_image_path):
    print(f"Using user-provided image: {user_image_path}")
    image_to_process = user_image_path
else:
    print(f"User image '{user_image_path}' not found.")
    print(f"Attempting to download sample image from {default_image_url}...")
    try:
        response = requests.get(default_image_url, headers=headers, stream=True, timeout=15)
        response.raise_for_status()
        with open(downloaded_image_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Sample image downloaded successfully as {downloaded_image_path}")
        image_to_process = downloaded_image_path
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

# --- Candidate Labels Definition ---
# 4. Define the candidate text labels to classify the image against.
#    *** Change these labels to be relevant to YOUR image! ***
candidate_labels = ["cats sleeping", "dogs playing fetch", "remote control", "city buildings", "beach scene", "furniture"]

print(f"\nUsing image: {os.path.basename(image_to_process)}")
print(f"Candidate Labels: {candidate_labels}")
# ---------------------------------


# --- Model Loading ---
print("\nLoading Zero-Shot Image Classification model (may download on first run)...")
try:
    # Use the "zero-shot-image-classification" pipeline task
    classifier = pipeline(
        "zero-shot-image-classification",
        model="openai/clip-vit-base-patch32", # Explicit CLIP model name
        device=0 if torch.cuda.is_available() else -1 # Use GPU if available, else CPU
        )
    print("Model loaded successfully.")
    if torch.cuda.is_available():
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure relevant libraries are installed: transformers, torch, Pillow, ftfy, regex...")
    exit()
# ----------------------

# --- Zero-Shot Image Classification ---
print("\nClassifying image against candidate labels...")
try:
    # Pass the image path and the list of candidate labels
    results = classifier(image_to_process, candidate_labels=candidate_labels)
    print("Classification complete.")

    # 6. Print the results
    #    The result is a list of dictionaries, one for each label,
    #    sorted by score (highest relevance first).
    print("\n--- Zero-Shot Classification Results ---")
    if not results:
        print("Model could not classify the image against labels.")
    else:
        for i, prediction in enumerate(results):
            label = prediction['label']
            score = prediction['score'] # Score indicates relevance/similarity
            print(f"Rank {i+1}: Score: {score:.4f}, Label: {label}")
    print("---------------------------------------")

except Exception as e:
    print(f"Error during Zero-Shot Image Classification: {e}")
    if isinstance(e, FileNotFoundError):
         print(f"Internal error: Could not find the image file at {image_to_process}")

# ------------------------------------

print("\nExample finished.")