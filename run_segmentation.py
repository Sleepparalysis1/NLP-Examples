# Import necessary libraries
from transformers import pipeline
import torch
import os
import requests
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import random # For generating random colors

print("-------------------------------------------")
print("Hugging Face Local Inference Example")
print("Task: Image Segmentation (Semantic)")
print("Model: nvidia/segformer-b0-finetuned-ade-512-512")
print("-------------------------------------------")

# --- Image Handling ---
# 1. Define a path where the user *might* place their own image.
user_image_path = "my_segmentation_image.jpg" # <-- YOU CAN CHANGE THIS

# 2. Define the same default image URL (city street view) and download path.
#    Source: https://unsplash.com/photos/t倭YF547b0 (by Charles POSTIAUX)
default_image_url = "https://unsplash.com/photos/t倭YF547b0/download?force=true&w=640"
downloaded_image_path = "segmentation_sample_image.jpg"
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

# --- Model Loading ---
print("\nLoading Image Segmentation model (may download on first run)...")
try:
    # Use the "image-segmentation" pipeline task
    segmenter = pipeline(
        "image-segmentation",
        model="nvidia/segformer-b0-finetuned-ade-512-512", # Explicit SegFormer model
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

# --- Image Segmentation ---
print(f"\nPerforming segmentation for '{os.path.basename(image_to_process)}'...")
segments = [] # Initialize segments list
try:
    # The pipeline returns a list of dicts, each with score, label, mask
    segments = segmenter(image_to_process)
    print("Segmentation complete.")

except Exception as e:
    print(f"Error during image segmentation: {e}")
    if isinstance(e, FileNotFoundError):
         print(f"Internal error: Could not find the image file at {image_to_process}")
    exit() # Exit if segmentation fails

# --- Visualization ---
print("\nGenerating visualization...")
output_visualization_path = "segmentation_visualization.png"
try:
    # Load the original image again for visualization
    original_image = Image.open(image_to_process).convert("RGBA") # Use RGBA for alpha blending
    width, height = original_image.size

    # Create a matplotlib figure
    plt.style.use('default') # Use default style
    fig, ax = plt.subplots(1, figsize=(10 * (width/height), 10) if height > 0 else (10, 10)) # Adjust figsize aspect ratio
    ax.imshow(original_image) # Display original image

    # Store colors assigned to labels for legend
    color_map = {}
    legend_elements = []

    print(f"Processing {len(segments)} detected segments...")
    for segment in segments:
        label = segment['label']
        mask_pil = segment['mask'] # This is a PIL Image mask

        # Assign a random color if label is new, otherwise use existing color
        if label not in color_map:
            # Generate a random RGBA color with some transparency
            color = tuple(np.random.rand(3)) # RGB tuple (0-1 range)
            color_map[label] = (*color, 0.5) # Store RGBA with 0.5 alpha
            # Create a patch for the legend
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=color_map[label], edgecolor='none', label=label))


        # Convert PIL mask to NumPy array to work with channels
        mask_np = np.array(mask_pil) # Shape (height, width)

        # Create an RGBA image for the colored mask overlay
        colored_mask_overlay = np.zeros((height, width, 4), dtype=np.float32) # Use float for alpha blending

        # Apply the color and alpha only where the mask is active (non-zero)
        mask_indices = mask_np > 0
        colored_mask_overlay[mask_indices, :3] = color_map[label][:3] # Apply RGB
        colored_mask_overlay[mask_indices, 3] = color_map[label][3]  # Apply Alpha

        # Overlay the colored mask onto the plot
        ax.imshow(colored_mask_overlay)

    # Add legend if segments were found
    if legend_elements:
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    ax.axis('off') # Hide axes
    plt.tight_layout() # Adjust layout

    # Save the visualization
    plt.savefig(output_visualization_path, bbox_inches='tight', dpi=150) # Save with tight bounding box
    print(f"\n--- Output ---")
    print(f"Segmentation visualization saved successfully to: {output_visualization_path}")
    print("--------------")
    # plt.show() # Optional: display plot if in interactive environment

except ImportError:
     print("Error: Matplotlib or NumPy not found. Cannot create visualization.")
     print("Please install them: pip install matplotlib numpy")
except FileNotFoundError:
     print(f"Error: Could not open image file for drawing: {image_to_process}")
except Exception as e:
    print(f"Error during visualization or saving: {e}")

# ---------------------

print("\nExample finished.")