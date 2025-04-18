# Import necessary libraries
from transformers import pipeline
import torch
import pandas as pd
import os

# Check for torch-scatter, often needed for TAPAS
try:
    import torch_scatter
    print("torch_scatter library found.")
except ImportError:
    print("Warning: torch-scatter library not found. Install it: pip install torch-scatter")
    print("TAPAS models might fail without it.")


print("-------------------------------------------")
print("Hugging Face Local Inference Example")
print("Task: Table Question Answering")
print("Model: google/tapas-base-finetuned-wtq (TAPAS)")
print("-------------------------------------------")

# --- USER ACTION REQUIRED: Define Table and Question ---

# 1. Define the table data as a dictionary
#    (Modify this data to represent your table)
#    Using sample activities relevant to Perth on a Friday afternoon/evening
data = {
    'Activity': ["Sunset Drinks", "Live Music", "Concert", "Dinner", "Late Night Coffee"],
    'Location': ["Cottesloe Beach Hotel", "The Bird (Northbridge)", "RAC Arena", "Sauma (Northbridge)", "Kafka Coffee Shop"],
    'Time': ["5:00 PM", "6:30 PM", "7:30 PM", "8:00 PM", "10:00 PM"],
    'Cost ($)': [25, 15, 120, 60, 8],
    'Day': ["Friday", "Friday", "Friday", "Friday", "Friday"]
}
# Convert the dictionary to a Pandas DataFrame - TAPAS pipeline expects this format
table = pd.DataFrame(data)

print("\nConverting table data to string type for model compatibility...")
table = table.astype(str)

# 2. Define the question about the table
#    *** Change this question to query your table data! ***
question = "What's the total cost?"
# question = "What time is the concert at RAC Arena?"
# Other examples: "How much does Dinner cost?", "Which activities are in Northbridge?"

# -------------------------------------------------------

print("\nTable Data (all columns as strings):")
print(table.to_string())

print(f"\nQuestion: \"{question}\"")


# --- Model Loading ---
print("\nLoading Table QA model (may download on first run)...")
try:
    # Use the "table-question-answering" pipeline task
    tqa_pipeline = pipeline(
        "table-question-answering",
        model="google/tapas-base-finetuned-wtq", # Explicit TAPAS model
        device=0 if torch.cuda.is_available() else -1 # Use GPU if available, else CPU
        )
    print("Model loaded successfully.")
    if torch.cuda.is_available():
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure 'transformers', 'torch', 'pandas', and 'torch-scatter' are installed.")
    exit()
# ----------------------

# --- Table Question Answering ---
print("\nAnswering question based on the table...")
try:
    # Pass the pandas DataFrame and the query string
    result = tqa_pipeline(table=table, query=question)
    print("Answer generation complete.")

    # 5. Print the result
    #    The result format includes 'answer', potentially 'coordinates' and 'cells'
    print("\n--- Predicted Answer ---")
    if result and 'answer' in result:
        print(f"Answer: {result['answer']}")
        # You can also inspect other parts of the result if needed:
        # print(f"Coordinates: {result.get('coordinates')}")
        # print(f"Cells: {result.get('cells')}")
    else:
        print("Could not determine an answer from the table.")
        print(f"Pipeline output: {result}") # Print full output for debugging
    print("------------------------")

except Exception as e:
    print(f"Error during Table Question Answering: {e}")
    print("Ensure the table is a valid Pandas DataFrame and the question relates to it.")

# ----------------------------

print("\nExample finished.")