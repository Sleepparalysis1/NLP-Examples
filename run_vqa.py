# Import necessary libraries
from transformers import pipeline
import torch
import os
import requests
from PIL import Image

print("-------------------------------------------")
print("Hugging Face Local Inference Example")
print("Task: Visual Question Answering (VQA)")
print("Model: dandelin/vilt-b32-finetuned-vqa (ViLT)")
print("-------------------------------------------")

# --- Image Handling ---
# 1. Define a path where the user *might* place their own image.
user_image_path = "cats.jpg" # <-- YOU CAN CHANGE THIS

# 2. Define a default image URL (COCO cats/remote) and download path.
default_image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
downloaded_image_path = "vqa_sample_image.jpg"
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

# --- Question Definition ---
# 4. Define the question about the image.
#    *** Change this question if you are using your own image! ***
question = "How many cats are in the picture?"
# Other questions for sample image: "What color is the couch?", "What is on the table?"

print(f"\nUsing image: {os.path.basename(image_to_process)}")
print(f"Asking question: \"{question}\"")
# ---------------------------


# --- Model Loading ---
print("\nLoading VQA model (may download on first run)...")
try:
    # Use the "visual-question-answering" pipeline task
    vqa_pipeline = pipeline(
        "visual-question-answering",
        model="dandelin/vilt-b32-finetuned-vqa", # Explicit ViLT model name
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

# --- Visual Question Answering ---
print("\nAnswering question based on the image...")
try:
    # Pass both the image (path or PIL object) and the question
    # top_k=1 returns the single most likely answer
    answers = vqa_pipeline(image=image_to_process, question=question, top_k=1)
    print("Answer generation complete.")

    # 6. Print the results
    #    The result is a list of dictionaries, each with 'answer' and 'score'
    print("\n--- Predicted Answer(s) ---")
    if not answers:
        print("Model could not determine an answer.")
    else:
        for i, answer_data in enumerate(answers):
            answer_text = answer_data.get('answer', 'Answer format unexpected')
            score = answer_data.get('score', 0.0) # Default score to 0.0 if missing
            print(f"Answer {i+1}:")
            print(f"  Text: \"{answer_text}\"")
            print(f"  Score: {score:.4f}")
            print("-" * 20) # Separator

    print("---------------------------")

except Exception as e:
    print(f"Error during Visual Question Answering: {e}")
    if isinstance(e, FileNotFoundError):
         print(f"Internal error: Could not find the image file at {image_to_process}")

# -----------------------

print("\nExample finished.")