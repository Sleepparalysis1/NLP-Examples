# Import pipeline and torch
from transformers import pipeline
import torch
import os

print("-------------------------------------------")
print("Hugging Face Local Inference Example")
print("Task: Part-of-Speech (POS) Tagging")
print("Model: vblagoje/bert-english-uncased-finetuned-pos")
print("-------------------------------------------")

# --- USER Configuration ---
# 1. Define the text you want to tag
#    Using context relevant to late Friday evening in Perth
text_to_tag = "Late Friday night in Perth. Thinking about getting some sleep soon, but the city lights look nice."

# --------------------------

# --- Model Loading ---
print("\nLoading POS Tagging model (may download on first run)...")
try:
    # Use the "token-classification" pipeline task
    # aggregation_strategy="simple" groups sub-word tokens (like ##ing) into whole words
    tagger = pipeline(
        "token-classification",
        model="vblagoje/bert-english-uncased-finetuned-pos",
        aggregation_strategy="simple", # Get word-level tags
        device=0 if torch.cuda.is_available() else -1
        )
    print("Model loaded successfully.")
    if torch.cuda.is_available():
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure 'transformers' and 'torch' are installed.")
    exit()
# ----------------------

# --- POS Tagging ---
print(f"\nInput Text:\n\"{text_to_tag}\"")
print("\nPerforming POS Tagging...")
pos_tags = []
try:
    # The pipeline returns a list of dictionaries for each token/word
    pos_tags = tagger(text_to_tag)
    print("Tagging complete.")

    # 4. Print the results
    print("\n--- POS Tagging Results ---")
    if not pos_tags:
        print("Could not extract POS tags.")
    else:
        # Find max word length for alignment
        max_len = 0
        if pos_tags:
             max_len = max(len(tag['word']) for tag in pos_tags) + 1 # Add padding

        for tag_data in pos_tags:
            word = tag_data['word']
            pos_tag = tag_data['entity_group'] # The pipeline puts the tag here for token classification
            score = tag_data['score']
            # Print aligned output
            print(f"Word: {word:<{max_len}} Tag: {pos_tag:<6} Score: {score:.4f}")

    print("-----------------------------")
    print("(Tags based on Penn Treebank Tagset, e.g., NNP=Proper Noun, NN=Noun, IN=Preposition, VBG=Verb Gerund/Participle)")


except Exception as e:
    print(f"Error during POS Tagging: {e}")

# -------------------

print("\nExample finished.")