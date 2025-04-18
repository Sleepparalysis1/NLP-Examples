# Import necessary libraries
from transformers import pipeline
import torch
import os

# Check for optional audio libraries
try:
    import soundfile as sf
    print(f"Soundfile library found.")
except ImportError:
    print("Warning: Soundfile library not found. Install it: pip install soundfile")
    print("Functionality might be limited depending on audio formats if using local files.")
try:
    import librosa
    print(f"Librosa library found.")
except ImportError:
     print("Warning: Librosa library not found. Install it: pip install librosa")
     print("Audio loading from datasets or certain formats might fail.")

print("-------------------------------------------")
print("Hugging Face Local Inference Example")
print("Task: Automatic Speech Recognition (ASR)")
print("Model: openai/whisper-base")
print("-------------------------------------------")
print("This example uses a user-provided audio file if found,")
print("otherwise it falls back to a sample from 'datasets'.")

# --- USER ACTION RECOMMENDED ---
# 1. Define the path where your local audio file MIGHT be.
#    If this file exists, the script will use it. Otherwise, it downloads a sample.
#    Replace placeholder or create a file with this name.
user_audio_path = "./asr_sample.wav" # <-- CHANGE THIS or place your file here

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
        # Using split='validation[:1]' to only load the first sample
        ds = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:1]")
        audio_sample = ds[0]["audio"]
        # The pipeline usually expects the dictionary format from datasets
        audio_input = audio_sample.copy()
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

# --- Model Loading (Same as before) ---
print("\nLoading ASR model (may download on first run)...")
try:
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",
        device=0 if torch.cuda.is_available() else -1
        )
    print("Model loaded successfully.")
    if torch.cuda.is_available():
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure 'transformers', 'torch', 'soundfile', 'librosa' are installed.")
    exit()
# --------------------------------------

# --- Transcription ---
print("\nPerforming speech-to-text transcription...")
try:
    # The pipeline should handle either the file path string
    # or the dictionary containing {'array': ..., 'sampling_rate': ...}
    transcription = pipe(audio_input)
    print("Transcription complete.")

    # 5. Print the result
    print("\n--- Transcription Result ---")
    if transcription and 'text' in transcription:
        print(f"Recognized Text: \"{transcription['text'].strip()}\"")
    else:
        print("Could not transcribe audio.")
        if isinstance(transcription, dict):
             print(f"Pipeline output: {transcription}")
    print("----------------------------")

except FileNotFoundError:
     # This error might occur if the user path existed initially but was removed,
     # or if the pipeline fails to resolve the path internally.
     print(f"ERROR: The audio file was not found by the pipeline at path: {audio_input if isinstance(audio_input, str) else 'loaded data'}")
except Exception as e:
    print(f"Error during transcription: {e}")
    if isinstance(audio_input, str):
        print("Ensure the audio file is a supported format (e.g., WAV, FLAC, MP3).")
        print("You might need 'ffmpeg' installed on your system for certain formats.")

# -----------------------

print("\nExample finished.")