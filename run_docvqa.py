# Import necessary libraries
from transformers import pipeline
import torch
import os
import requests
from PIL import Image

# Check for sentencepiece, required by Donut
try:
    import sentencepiece
    print("sentencepiece library found.")
except ImportError:
    print("Warning: sentencepiece library not found. Install it: pip install sentencepiece")
    print("Donut model may fail without it.")


print("-------------------------------------------")
print("Hugging Face Local Inference Example")
print("Task: Document Question Answering (DocVQA)")
print("Model: naver-clova-ix/donut-base-finetuned-docvqa (Donut)")
print("-------------------------------------------")

# --- Image Handling ---
# 1. Define a path where the user *might* place their own document image.
user_doc_image_path = "ReceiptSwiss.jpg" # <-- YOU CAN CHANGE THIS (e.g., my_receipt.jpg)

# 2. Define a default image URL (simple Swiss receipt) and download path.
#    Source: https://commons.wikimedia.org/wiki/File:ReceiptSwiss.jpg (CC BY-SA 4.0)
default_image_url = "https://upload.wikimedia.org/wikipedia/commons/0/0b/ReceiptSwiss.jpg"
downloaded_image_path = "docvqa_sample_receipt.jpg"
image_to_process = None
headers = {'User-Agent': 'Mozilla/5.0'} # Header for requests

# 3. Decide which image to use
if os.path.exists(user_doc_image_path):
    print(f"Using user-provided document image: {user_doc_image_path}")
    image_to_process = user_doc_image_path
else:
    print(f"User document image '{user_doc_image_path}' not found.")
    print(f"Attempting to download sample receipt image from {default_image_url}...")
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
    print("\nERROR: No valid document image available to process.")
    exit()
# ------------------------

# --- Question Definition ---
# 4. Define the question about the document image.
#    *** Change this question if you are using your own document image! ***
question = "What is the total amount?"
# Other questions for sample image: "What is the date?", "What items were purchased?"

print(f"\nUsing document image: {os.path.basename(image_to_process)}")
print(f"Asking question: \"{question}\"")
# ---------------------------


# --- Model Loading ---
print("\nLoading DocVQA model (may download on first run)...")
try:
    # Use the "document-question-answering" pipeline task
    # Note: This task might require specific versions or setups.
    # If issues arise, check transformers documentation for Donut usage.
    doc_qa_pipeline = pipeline(
        "document-question-answering",
        model="naver-clova-ix/donut-base-finetuned-docvqa", # Explicit Donut model
        device=0 if torch.cuda.is_available() else -1
        )
    print("Model loaded successfully.")
    if torch.cuda.is_available():
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU.")
except ValueError as e:
     # Catch specific error if task isn't registered maybe
     print(f"Error loading pipeline: {e}")
     print("Ensure you have a compatible transformers version and all dependencies.")
     print("Dependencies might include: transformers, torch, Pillow, sentencepiece, torchvision, timm")
     exit()
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure relevant libraries are installed.")
    exit()
# ----------------------

# --- Document Question Answering ---
print("\nAnswering question based on the document image...")
try:
    # Pass the image path and the question
    # Requesting top_k=1 for the most likely answer
    answers = doc_qa_pipeline(image=image_to_process, question=question, top_k=1)
    print("Answer generation complete.")

    # 6. Print the results
    #    The result is usually a list of dictionaries, each with 'answer' and 'score'
    print("\n--- Predicted Answer(s) ---")
    if not answers:
        print("Model could not determine an answer.")
    else:
        for i, answer_data in enumerate(answers):
            answer_text = answer_data.get('answer', 'Answer format unexpected')
            score = answer_data.get('score', 0.0)
            print(f"Answer {i+1}:")
            print(f"  Text: \"{answer_text}\"")
            print(f"  Score: {score:.4f}")
            print("-" * 20) # Separator

    print("---------------------------")

except Exception as e:
    print(f"Error during Document Question Answering: {e}")
    if isinstance(e, FileNotFoundError):
         print(f"Internal error: Could not find the image file at {image_to_process}")
    print("Ensure document image is clear and question is relevant.")

# -----------------------------

print("\nExample finished.")