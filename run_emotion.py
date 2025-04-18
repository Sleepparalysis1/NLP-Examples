# Import the pipeline function and torch
from transformers import pipeline
import torch
import operator # To sort results later

print("-------------------------------------------")
print("Hugging Face Local Inference Example")
print("Task: Text Classification (Emotion Detection)")
print("Model: j-hartmann/emotion-english-distilroberta-base")
print("-------------------------------------------")

# 1. Load the text classification pipeline, explicitly specifying the emotion model
print("Loading emotion classification model (may download on first run)...")
try:
    # Use the "text-classification" pipeline task
    emotion_pipeline = pipeline(
        "text-classification", # Standard task name
        model="j-hartmann/emotion-english-distilroberta-base", # Explicit model name
        tokenizer="j-hartmann/emotion-english-distilroberta-base",
        # Get scores for all labels, not just the top one
        return_all_scores=True,
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

# 2. Define some sentences to classify for emotion
#    Using context relevant to Friday afternoon in Perth
sentences_to_classify = [
    "It's a beautiful sunny Friday afternoon in Perth, feeling wonderful!",
    "This traffic jam getting out of the city is absolutely infuriating.",
    "Slightly anxious about finishing this important work before the weekend.",
    "Just noticed my train home might be delayed, how annoying.",
    "Looking forward to relaxing by the Swan River later, should be peaceful.",
    "The news report about the freeway accident is quite distressing.",
    "Perfect weather today for a walk in Kings Park." # Might be neutral or joy
]

print("\nSentences to Classify for Emotion:")
for i, s in enumerate(sentences_to_classify):
    print(f"{i+1}. \"{s}\"")


# 3. Run the emotion classification pipeline
print("\nClassifying emotions...")
try:
    # The pipeline returns a list of lists (one list per sentence)
    # Each inner list contains dictionaries {'label': emotion, 'score': probability}
    results = emotion_pipeline(sentences_to_classify)
    print("Classification complete.")

    # 4. Print the results for each sentence
    print("\n--- Emotion Classification Results ---")
    if not results:
        print("Could not classify emotions.")
    else:
        # Iterate through sentences and their corresponding results
        for i, sentence_results in enumerate(results):
            print(f"\nSentence {i+1}: \"{sentences_to_classify[i]}\"")

            # Sort the results by score in descending order
            sorted_results = sorted(sentence_results, key=operator.itemgetter('score'), reverse=True)

            print("  Predicted Emotions (Top First):")
            for prediction in sorted_results:
                label = prediction['label']
                score = prediction['score']
                print(f"    - {label.capitalize()}: {score:.4f}")
            print("-" * 20) # Separator

    print("------------------------------------")

except Exception as e:
    print(f"Error during emotion classification: {e}")


print("\nExample finished.")