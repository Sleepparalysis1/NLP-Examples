# Import the pipeline function and torch
from transformers import pipeline
import torch

print("-------------------------------------------")
print("Hugging Face Local Inference Example")
print("Task: Named Entity Recognition (NER)")
print("Model: dbmdz/bert-large-cased-finetuned-conll03-english")
print("-------------------------------------------")

# 1. Load the NER pipeline, explicitly specifying the model
print("Loading NER model (may download on first run)...")
try:
    ner_pipeline = pipeline(
        "ner", # Specify the task
        model="dbmdz/bert-large-cased-finetuned-conll03-english", # Explicit model name
        # Use an aggregation strategy to group recognized parts of the same entity
        # 'simple' groups consecutive tokens with the same entity type (e.g., "Elizabeth", "Quay" -> "Elizabeth Quay")
        aggregation_strategy="simple",
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

# 2. Define the text you want to analyze
#    Using context relevant to Friday afternoon in Perth
text_to_analyze = """
It was shaping up to be a busy Friday afternoon for Ms. Eleanor Vance at the Perth branch of Globex Corporation.
She needed to finalize the proposal for the new project near Yagan Square before meeting Mr. David Chen from Zenith Solutions later today.
Maybe afterwards, she could quickly visit the Art Gallery of Western Australia.
"""

print(f"\nText to Analyze:\n\"{text_to_analyze}\"")


# 3. Run the NER pipeline
print("\nPerforming Named Entity Recognition...")
try:
    ner_results = ner_pipeline(text_to_analyze)
    print("NER complete.")

    # 4. Print the results
    print("\n--- Identified Entities ---")
    if not ner_results:
        print("No entities found.")
    else:
        for entity in ner_results:
            # The 'simple' aggregation strategy provides these keys:
            entity_type = entity['entity_group']
            confidence_score = entity['score']
            entity_text = entity['word']

            # Map common entity types for clarity (based on CoNLL-2003)
            # I-PER/B-PER -> PER (Person)
            # I-ORG/B-ORG -> ORG (Organization)
            # I-LOC/B-LOC -> LOC (Location)
            # I-MISC/B-MISC -> MISC (Miscellaneous)
            # The pipeline with aggregation_strategy already simplifies this to PER, ORG, LOC, MISC

            print(f"  - Entity: \"{entity_text}\"")
            print(f"    Type: {entity_type}")
            print(f"    Confidence: {confidence_score:.4f}")
            print("-" * 15) # Separator

    print("-------------------------")

except Exception as e:
    print(f"Error during NER: {e}")


print("\nExample finished.")