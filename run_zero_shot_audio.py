# Import necessary libraries
from transformers import pipeline
import torch
import os
import datasets
import numpy as np # Added numpy

# Check for required audio libraries
try:
    import soundfile as sf
    print(f"Soundfile library found.")
except ImportError:
    print("ERROR: Soundfile library not found. Install it: pip install soundfile")
    print("You might also need system dependencies: sudo apt update && sudo apt install libsndfile1")
    exit()
try:
    import librosa
    print(f"Librosa library found.")
except ImportError:
     print("ERROR: Librosa library not found. Install it: pip install librosa or transformers[audio]")
     exit()


print("-------------------------------------------")
print("Hugging Face Local Inference Example")
print("Task: Zero-Shot Audio Classification")
print("Model: laion/clap-htsat-unfused (CLAP)")
print("-------------------------------------------")
print("This example uses a user-provided audio file if found,")
print("otherwise it falls back to a sample from 'datasets'.")

# --- USER ACTION RECOMMENDED ---
# 1. Define the path where your local audio file MIGHT be.
user_audio_path = "tts_output.wav" # <-- CHANGE THIS

# 2. Define the candidate text labels to classify the audio against.
candidate_labels = ["someone speaking", "techno music", "dog barking", "car passing by", "silence", "waves crashing", "male voice", "female voice"]
# ----------------------------------

# --- Determine and Load Audio Input ---
audio_input_np = None # Will hold the numpy array
input_source_message = ""
target_sr = 48000 # CLAP models often expect 48kHz, check model card if unsure

# Function to load and preprocess audio
def load_and_preprocess_audio(path, target_sr):
    try:
        audio_data, sample_rate = sf.read(path, dtype='float32')
        print(f"Loaded audio from {path} (Sample Rate: {sample_rate} Hz, Shape: {audio_data.shape})")

        # Convert to mono if necessary
        if audio_data.ndim > 1:
            print("Audio is not mono, converting to mono by averaging channels...")
            audio_data = np.mean(audio_data, axis=1)

        # Resample if necessary
        if sample_rate != target_sr:
            print(f"Resampling from {sample_rate} Hz to {target_sr} Hz...")
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)
            print(f"Resampled audio shape: {audio_data.shape}")

        return audio_data, target_sr
    except Exception as e:
        print(f"Error loading or processing audio file {path}: {e}")
        return None, None

# Check if the user's audio file exists
if os.path.exists(user_audio_path):
    print(f"\nFound user-provided audio file.")
    audio_input_np, sr = load_and_preprocess_audio(user_audio_path, target_sr)
    if audio_input_np is not None:
        input_source_message = f"Using user-provided audio file: {user_audio_path} (Processed to {sr} Hz Mono)"
    # If loading user file failed, audio_input_np is None, fallback will trigger
else:
    print(f"\nUser audio file not found at: {user_audio_path}")

# If user file wasn't found or failed to load, try loading the dataset sample
if audio_input_np is None:
    print("Attempting to load sample audio using 'datasets' library...")
    try:
        import datasets
        print("Loading sample audio data (may download on first run)...")
        ds = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:1]")
        audio_sample = ds[0]["audio"]
        audio_data = audio_sample["array"].astype(np.float32) # Extract array ensure float32
        sample_rate = audio_sample["sampling_rate"]
        print(f"Loaded sample (Sample Rate: {sample_rate} Hz, Shape: {audio_data.shape})")

        # Convert to mono (likely already mono, but check)
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Resample if necessary
        if sample_rate != target_sr:
            print(f"Resampling sample from {sample_rate} Hz to {target_sr} Hz...")
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)
            print(f"Resampled sample shape: {audio_data.shape}")

        audio_input_np = audio_data
        input_source_message = f"Using sample audio (Processed to {target_sr} Hz Mono from librispeech_asr_dummy)"
        print("Sample audio loaded and processed successfully.")

    except ImportError:
        print("\nERROR: 'datasets' library not found. Cannot download sample.")
    except Exception as e:
        print(f"\nERROR: Failed to load or process sample audio dataset: {e}")

# Verify that we have a valid audio input source
if audio_input_np is None:
    print("\nError: Could not find or load any audio input. Exiting.")
    exit()
else:
    print(f"\n{input_source_message}")
# --------------------------------------

# --- Model Loading (Same as before) ---
print("\nLoading Zero-Shot Audio Classification model (may download on first run)...")
try:
    classifier = pipeline(
        "zero-shot-audio-classification",
        model="laion/clap-htsat-unfused",
        device=0 if torch.cuda.is_available() else -1
        )
    print("Model loaded successfully.")
    # ... (rest of device printing) ...
except Exception as e:
    print(f"Error loading model: {e}")
    exit()
# --------------------------------------

# --- Zero-Shot Audio Classification ---
print(f"\nClassifying audio against labels: {candidate_labels}")
try:
    # Pass the processed NumPy array containing the audio waveform
    results = classifier(audio_input_np, candidate_labels=candidate_labels)
    print("Classification complete.")

    # Print the results (Same as before)
    print("\n--- Zero-Shot Audio Classification Results ---")
    if not results:
        print("Could not classify the audio.")
    else:
        for i, prediction in enumerate(results):
            label = prediction['label']
            score = prediction['score']
            print(f"Rank {i+1}: Score: {score:.4f}, Label: {label}")
    print("------------------------------------------")

except Exception as e:
    print(f"Error during zero-shot audio classification: {e}")
    print("Ensure audio data was loaded correctly and model is compatible.")

# ------------------------------------

print("\nExample finished.")