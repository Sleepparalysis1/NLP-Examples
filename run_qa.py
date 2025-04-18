# Import the pipeline function and torch
from transformers import pipeline
import torch

print("-------------------------------------------")
print("Hugging Face Local Inference Example")
print("Task: Question Answering (Extractive)")
print("Model: distilbert-base-cased-distilled-squad")
print("-------------------------------------------")

# 1. Load the Question Answering pipeline, explicitly specifying the model
print("Loading QA model (may download on first run)...")
try:
    qa_pipeline = pipeline(
        "question-answering", # Specify the task
        model="distilbert-base-cased-distilled-squad", # Explicit model name
        tokenizer="distilbert-base-cased-distilled-squad", # Often good to specify tokenizer too
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

# 2. Define the context paragraph containing the information
#    Using context relevant to Friday afternoon in Perth
context = """
It's Friday afternoon, just after 2:00 PM here in Perth, Western Australia. The sun is shining brightly.
Many downtown office workers are wrapping up their tasks, anticipating the weekend. Popular after-work destinations
include waterfront bars at Elizabeth Quay or heading towards Cottesloe Beach for the sunset later on.
Traffic is building up; major roadworks on the Graham Farmer Freeway contribute to delays for those heading east.
Meanwhile, in the central business district, setup crews are busy arranging light installations for the upcoming 'Lumiere Perth' festival, scheduled to illuminate the city starting next Friday night.
Security personnel are also visibly present around the installation zones near Forrest Place.
"""

# 3. Define the question you want to ask about the context
question = "What event is causing setup crews to be busy in the central business district?"
# Another possible question: "Where might people go after work?"
# Another possible question: "What road has major roadworks?"

print(f"\nContext:\n\"{context}\"")
print(f"\nQuestion: {question}")


# 4. Run the QA pipeline
#    You pass both the question and the context.
print("\nFinding answer in context...")
try:
    qa_result = qa_pipeline(question=question, context=context)
    print("Answer extraction complete.")

    # 5. Print the result
    #    The result is a dictionary containing the answer span, score, start/end indices.
    print("\n--- Extracted Answer ---")
    if qa_result:
        print(f"Answer:   \"{qa_result['answer']}\"")
        print(f"Confidence: {qa_result['score']:.4f}")
        # print(f"Start Index: {qa_result['start']}") # Character index in context
        # print(f"End Index:   {qa_result['end']}")   # Character index in context
    else:
        print("Could not find an answer in the context.")
    print("------------------------")

except Exception as e:
    print(f"Error during Question Answering: {e}")


print("\nExample finished.")