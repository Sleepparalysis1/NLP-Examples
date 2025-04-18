# Import the pipeline function and torch
from transformers import pipeline
import torch

print("-------------------------------------------")
print("Hugging Face Local Inference Example")
print("Task: Zero-Shot Classification")
print("-------------------------------------------")
print("This classifies text using labels the model wasn't necessarily trained on.")

# 1. Load the zero-shot classification pipeline
#    - Uses a suitable NLI model by default (e.g., BART fine-tuned on MNLI).
#    - Downloads and caches the model on the first run.
print("Loading model (may download on first run)...")
try:
    # The default model often works well, e.g., facebook/bart-large-mnli
    # You could specify another one compatible with zero-shot if needed.
    classifier = pipeline(
        "zero-shot-classification",
        # model="facebook/bart-large-mnli", # Often the default
        device=0 if torch.cuda.is_available() else -1 # Use GPU if available, else CPU
        )
    print("Model loaded successfully.")
    if torch.cuda.is_available():
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU.")

except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'transformers', 'sentencepiece', and 'torch' (or 'tensorflow') are installed.")
    exit()

# 2. Define the text sequence you want to classify
#    Let's use something relevant to your current context in Perth.
sequence_to_classify = """
It's Friday morning here in Perth, and the April weather forecast looks fantastic for the whole weekend ahead.
I should probably focus on finishing this quarterly financial report for work before lunchtime, but my mind keeps wandering.
I'm already picturing heading down to Scarborough Beach later this afternoon to catch some sun.
Tomorrow is trickier - maybe check out the Fremantle Markets for some local crafts and food, or perhaps undertake the longer drive down south towards the Margaret River region for some potential wine tasting?
That definitely involves more logistical travel planning and booking accommodation, though. Need to make a decision soon!
"""

# 3. Define the candidate labels you want to classify the sequence into
#    These can be anything you want!
candidate_labels = [
    'Business Meeting',
    'Leisure Activity',
    'Urgent Problem',
    'Travel Planning',
    'Scientific Research',
    'Outdoor Recreation'
]

print(f"\nSequence to classify: \"{sequence_to_classify}\"")
print(f"Candidate Labels: {candidate_labels}")

# 4. Run the classification
#    - By default, it assumes only one label is correct (multi_label=False).
#    - Set multi_label=True if the text could belong to multiple categories.
print("\nClassifying...")
try:
    results = classifier(
        sequence_to_classify,
        candidate_labels,
        # multi_label=True # Set this if text can fit multiple labels
    )
    print("Classification complete.")

    # 5. Print the results
    #    The results contain the labels sorted by their scores (highest first)
    print("\n--- Results ---")
    print(f"Sequence: \"{results['sequence']}\"")
    print("Scores per label:")
    for label, score in zip(results['labels'], results['scores']):
        print(f"  - {label}: {score:.4f}") # Format score to 4 decimal places
    print("----------------------")

except Exception as e:
    print(f"Error during classification: {e}")

print("\nExample finished.")