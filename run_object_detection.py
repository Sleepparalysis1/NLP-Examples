# Import pipeline, torch, os, requests, and Image/Draw/Font from PIL
from transformers import pipeline
import torch
import os
import requests
from PIL import Image, ImageDraw, ImageFont

print("-------------------------------------------")
print("Hugging Face Local Inference Example")
print("Task: Object Detection (with Annotation)")
print("Model: facebook/detr-resnet-50 (DETR)")
print("-------------------------------------------")

# --- Image Handling (Same as before) ---
user_image_path = "./obj_detection.JPG" # <-- YOU CAN CHANGE THIS
default_image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
downloaded_image_path = "detr_sample_image.jpg"
image_to_process = None

if os.path.exists(user_image_path):
    print(f"Using user-provided image: {user_image_path}")
    image_to_process = user_image_path
else:
    print(f"User image '{user_image_path}' not found.")
    print(f"Attempting to download sample image from {default_image_url}...")
    try:
        response = requests.get(default_image_url, stream=True)
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
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Could not download sample image: {e}")
    except IOError as e:
        print(f"ERROR: Could not save downloaded image: {e}")

if image_to_process is None:
    print("\nERROR: No valid image available to process.")
    exit()
# --------------------------------------------------

# --- Model Loading (Same as before) ---
print("\nLoading object detection model (may download on first run)...")
try:
    object_detector = pipeline(
        "object-detection",
        model="facebook/detr-resnet-50",
        device=0 if torch.cuda.is_available() else -1
        )
    print("Model loaded successfully.")
    if torch.cuda.is_available():
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()
# --------------------------------------

# --- Object Detection (Same as before) ---
detection_threshold = 0.9
print(f"\nDetecting objects in '{os.path.basename(image_to_process)}' (Threshold: {detection_threshold})...")
detections = [] # Initialize detections list
try:
    detections = object_detector(image_to_process, threshold=detection_threshold)
    print("Object detection complete.")
except Exception as e:
    print(f"Error during object detection: {e}")
    exit()
# -----------------------------------------

# --- Annotation and Saving --- NEW SECTION ---
print("\nAnnotating image with detected objects...")
output_path = "object_detection_output.jpg"

try:
    # Open the original image to draw on
    # Convert to RGB ensure compatibility with drawing and saving as JPG
    img = Image.open(image_to_process).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Try to load a nicer font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 15) # You might need to install arial or change path
    except IOError:
        print("Arial font not found, using default font.")
        font = ImageFont.load_default()

    if not detections:
        print("No objects detected above the threshold to annotate.")
    else:
        print(f"Found {len(detections)} objects above threshold {detection_threshold}.")
        for detection in detections:
            label = detection['label']
            score = detection['score']
            box = detection['box']
            box_coords = (box['xmin'], box['ymin'], box['xmax'], box['ymax'])

            # Draw bounding box
            draw.rectangle(box_coords, outline="lime", width=3)

            # Prepare text label
            text = f"{label}: {score:.2f}"

            # Calculate text position (slightly above the box)
            # Use textbbox to get actual text size for better placement/background
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            text_x = box['xmin']
            text_y = box['ymin'] - text_height - 5 # Position text above box
            # Ensure text stays within image bounds (top)
            if text_y < 0:
                text_y = box['ymin'] + 2 # Place inside box if too close to top

            # Draw a filled rectangle background for the text for better visibility
            text_bg_coords = (text_x - 1, text_y - 1, text_x + text_width + 1, text_y + text_height + 1)
            draw.rectangle(text_bg_coords, fill="lime")

            # Draw the text
            draw.text((text_x, text_y), text, fill="black", font=font)

    # Save the annotated image
    img.save(output_path)
    print(f"\nAnnotated image saved successfully to: {output_path}")

    # Optional: Display the image (might not work on all systems/environments)
    # try:
    #     img.show()
    # except Exception as e:
    #     print(f"Could not display image automatically: {e}")

except FileNotFoundError:
    print(f"ERROR: Could not open image file for drawing: {image_to_process}")
except Exception as e:
    print(f"Error during image annotation or saving: {e}")

# --------------------------------------------------

print("\nExample finished.")