# Import necessary libraries
from transformers import pipeline
import torch
import os
import datasets

# Check for optional audio libraries
try:
    import soundfile as sf
    print(f"Soundfile library found.")
except ImportError:
    print("Warning: Soundfile library not found. Install it: pip install soundfile")
try:
    import librosa
    print(f"Librosa library found.")
except ImportError:
     print("Warning: Librosa library not found. Install it: pip install librosa")
try:
    import torchaudio
    print(f"Torchaudio library found.")
except ImportError:
     print("Warning: Torchaudio library not found. Install it: pip install torchaudio")


print("-------------------------------------------")
print("Hugging Face Local Inference Example")
print("Task: Audio Classification")
print("Model: MIT/ast-finetuned-audioset-10-10-0.4593 (AST)")
print("-------------------------------------------")
print("This example uses a user-provided audio file if found,")
print("otherwise it falls back to a sample from 'datasets'.")

# --- USER ACTION RECOMMENDED ---
# 1. Define the path where your local audio file MIGHT be.
user_audio_path = "my_audio_for_classification.wav" # <-- CHANGE THIS

# ----------------------------

# --- Determine Audio Input ---
audio_input = None
input_source_message = ""

# 2. Check if the user's audio file exists
if os.path.exists(user_audio_path):
    print(f"\nFound user-provided audio file.")
    audio_input = user_audio_path # Pipeline can handle the path directly
    input_source_message = f"Using user-provided audio file: {user_audio_path}"
else:
    # 3. If user file doesn't exist, try loading the dataset sample
    print(f"\nUser audio file not found at: {user_audio_path}")
    print("Attempting to load sample audio using 'datasets' library...")
    try:
        # Import datasets only if needed
        import datasets
        print("Loading sample audio data (may download on first run)...")
        ds = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:1]")
        audio_sample = ds[0]["audio"]
        # Pipeline can usually handle the dictionary or the raw array
        audio_input = audio_sample # Keep dict for potential metadata use by pipeline
        # Or potentially: audio_input = audio_sample["array"] # If pipeline prefers raw array
        input_source_message = f"Using sample audio (Rate: {audio_sample['sampling_rate']} Hz from librispeech_asr_dummy)"
        print("Sample audio loaded successfully.")
    except ImportError:
        print("\nERROR: 'datasets' library not found. Cannot download sample.")
        print("Please install it ('pip install datasets') or provide a valid local audio file.")
    except Exception as e:
        print(f"\nERROR: Failed to load sample audio dataset: {e}")
        print("Check internet connection or try providing a local audio file.")

# 4. Verify that we have a valid audio input source
if audio_input is None:
    print("\nError: Could not find or load any audio input. Exiting.")
    exit()
else:
    print(f"\n{input_source_message}")

# --- Model Loading ---
print("\nLoading Audio Classification model (may download on first run)...")
try:
    # Use the "audio-classification" pipeline task
    audio_classifier = pipeline(
        "audio-classification",
        model="MIT/ast-finetuned-audioset-10-10-0.4593", # Explicit AST model
        device=0 if torch.cuda.is_available() else -1
        )
    print("Model loaded successfully.")
    if torch.cuda.is_available():
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure relevant libraries are installed: transformers, torch, torchaudio, librosa, soundfile...")
    exit()
# ----------------------

# --- Audio Classification ---
print("\nClassifying audio...")
try:
    # Pass the audio input (path string or dict/array) to the pipeline
    # Get the top 5 predictions
    results = audio_classifier(audio_input, top_k=5)
    print("Classification complete.")

    # 5. Print the results
    #    The result is a list of dictionaries, each with 'label' and 'score'
    print(f"\n--- Top {len(results)} Audio Classification Predictions ---")
    if not results:
        print("Could not classify the audio.")
    else:
        for i, prediction in enumerate(results):
            label = prediction['label']
            score = prediction['score']
            print(f"Rank {i+1}: Score: {score:.4f}, Label: {label}")
    print("------------------------------------------")

except Exception as e:
    print(f"Error during audio classification: {e}")
    if isinstance(audio_input, str) and not os.path.exists(audio_input):
         print(f"Internal error: Could not find the audio file at {audio_input}")
    print("Ensure audio file is valid and required libraries (inc. ffmpeg if needed) are installed.")

# --------------------------

print("\nExample finished.")