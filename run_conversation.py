# Import pipeline, AutoTokenizer, and torch
from transformers import pipeline, AutoTokenizer
import torch
import os

print("-------------------------------------------")
print("Hugging Face Local Inference Example")
print("Task: Dialogue Simulation (via Text Generation)")
print("Model: microsoft/DialoGPT-medium")
print("Note: Using text-generation pipeline with manual history.")
print("-------------------------------------------")

# Define Model name
model_name = "microsoft/DialoGPT-medium"

# --- Model and Tokenizer Loading ---
print("\nLoading model and tokenizer (may download on first run)...")
try:
    # Load the text-generation pipeline
    generator = pipeline(
        "text-generation",
        model=model_name,
        device=0 if torch.cuda.is_available() else -1
    )
    # Load the tokenizer separately to access eos_token
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model and tokenizer loaded successfully.")
    if torch.cuda.is_available():
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU.")

except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    print("Ensure 'transformers' and 'torch' are installed.")
    exit()
# ------------------------------------

# --- Define Conversation Turns ---
user_inputs = [
    "Hi, how are you?",
    "What's a good plan for a Friday evening in Perth?", # Using context
    "That's okay. What do people usually do on Friday evenings?" # Follow-up
]

# Initialize dialogue history string
dialogue_history_string = ""
# --------------------------------

# --- Simulate Conversation ---
print("\n--- Starting Dialogue Simulation ---")
try:
    for i, user_text in enumerate(user_inputs):
        print(f"\nUser >>> {user_text}")

        # Construct the prompt by appending the new user input and EOS token
        # The EOS token signals the end of a turn for DialoGPT
        prompt = dialogue_history_string + user_text + tokenizer.eos_token

        # Generate response using the text-generation pipeline
        # We need to specify max_new_tokens to limit the response length
        # pad_token_id is often needed to suppress warnings during generation
        generated_sequences = generator(
            prompt,
            max_new_tokens=60,  # Adjust max response length as needed
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True, # Add some randomness
            temperature=0.7,
            top_k=50
        )

        # Extract the generated text from the result
        full_generated_text = generated_sequences[0]['generated_text']

        # Extract *only* the newly generated response part
        # Find the end of our prompt in the generated text and take the rest
        # Need to be careful here as the model might slightly reformat prompt internally
        # A simple way is to take text after the length of the prompt
        response_text = full_generated_text[len(prompt):].strip()

        # Handle empty responses (e.g., if only EOS token was generated)
        if not response_text:
            response_text = "(Model generated empty response)"


        print(f"Bot >>> {response_text}")
        print("--------------------")

        # Update the dialogue history string for the next turn
        dialogue_history_string = full_generated_text + tokenizer.eos_token


    print("\n--- Dialogue Finished ---")

except Exception as e:
    print(f"\nError during dialogue generation: {e}")

# ---------------------------

print("\nExample finished.")