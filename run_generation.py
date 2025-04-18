# Import the pipeline function and torch
from transformers import pipeline
import torch

print("-------------------------------------------")
print("Hugging Face Local Inference Example")
print("Task: Text Generation")
print("-------------------------------------------")

# 1. Load the text generation pipeline
#    - Uses GPT-2 model by default if 'model' isn't specified.
#    - Downloads and caches the model on the first run.
print("Loading model (may download on first run)...")
try:
    # You could specify a different model like "gpt2-medium", "distilgpt2", etc.
    generator = pipeline(
        "text-generation",
        model="gpt2", # Explicitly using the standard GPT-2 model
        device=0 if torch.cuda.is_available() else -1 # Use GPU if available, else CPU
        )
    print("Model loaded successfully.")
    if torch.cuda.is_available():
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU.")

except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'transformers' and 'torch' (or 'tensorflow') are installed.")
    exit()

# 2. Define the starting prompt for the text generation
#    Using context from the current time/location you provided.
prompt = f"Thinking about the beautiful weather in Perth on this Friday morning, the future of artificial intelligence seems"

print(f"\nStarting prompt: \"{prompt}...\"")

# 3. Define generation parameters
#    - max_length: The total length (prompt + generated text)
#    - num_return_sequences: How many different completions to generate
max_total_length = 100  # Generate text up to this total length
num_sequences = 1       # Generate one possible completion

print(f"Generating text (up to {max_total_length} tokens)...")

# 4. Run the generation process
try:
    generated_outputs = generator(
        prompt,
        max_length=max_total_length,
        num_return_sequences=num_sequences,
        # Common parameters you might add:
        # temperature=0.7, # Controls randomness (lower = more focused)
        # top_k=50,        # Considers only the top k likely next words
        # top_p=0.9,       # Considers words cumulative probability > p
        # no_repeat_ngram_size=2 # Prevents repeating sequences of N words
    )
    print("Generation complete.")

    # 5. Print the results
    print("\n--- Generated Text ---")
    for i, output in enumerate(generated_outputs):
        # The output dictionary contains the full generated text under 'generated_text'
        print(f"Result {i+1}:")
        print(output['generated_text'])
        print("-" * 20) # Separator
    print("----------------------")

except Exception as e:
    print(f"Error during generation: {e}")


print("\nExample finished.")