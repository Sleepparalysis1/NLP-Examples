# Import the pipeline function and torch
from transformers import pipeline
import torch

print("-------------------------------------------")
print("Hugging Face Local Inference Example")
print("Task: Fill-Mask (Masked Language Modeling)")
print("Model: roberta-base")
print("-------------------------------------------")

# 1. Load the fill-mask pipeline, explicitly specifying the RoBERTa model
print("Loading fill-mask model (may download on first run)...")
try:
    fill_mask_pipeline = pipeline(
        "fill-mask", # Specify the task
        model="roberta-base", # Explicit model name
        tokenizer="roberta-base", # Explicit tokenizer name
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

# 2. Define the text containing a mask token
#    IMPORTANT: RoBERTa uses '<mask>' as the mask token.
#    Using context relevant to Friday afternoon in Perth
text_with_mask = "Many people working in central Perth look forward to the <mask> at the end of a Friday afternoon around 2 PM."

print(f"\nInput Text with Mask:\n\"{text_with_mask}\"")


# 3. Run the fill-mask pipeline
#    We can ask for the top N predictions using top_k
num_predictions = 5
print(f"\nPredicting top {num_predictions} replacements for '<mask>'...")
try:
    # The pipeline predicts likely tokens to fill the mask
    predictions = fill_mask_pipeline(text_with_mask, top_k=num_predictions)
    print("Prediction complete.")

    # 4. Print the results
    #    The result is a list of dictionaries, each containing the predicted
    #    token string, the score, and the full sequence with the mask filled.
    print(f"\n--- Top {num_predictions} Predictions ---")
    if predictions:
        for i, prediction in enumerate(predictions):
            score = prediction['score']
            token_str = prediction['token_str'] # The predicted token (word part)
            sequence = prediction['sequence']   # The full sentence with the mask filled

            # Note: RoBERTa uses byte-level BPE, so token_str might start with 'Ġ'
            # representing a space, which is usually desired.
            print(f"{i+1}. Score: {score:.4f}")
            print(f"   Predicted Token: '{token_str}'")
            print(f"   Full Sentence: \"{sequence}\"")
            print("-" * 15) # Separator
    else:
        print("Could not generate predictions.")
    print("-------------------------")

except Exception as e:
    print(f"Error during fill-mask prediction: {e}")


print("\nExample finished.")