# Import necessary libraries
from transformers import pipeline, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import os
import datasets
import soundfile as sf

# Check for SpeechRecognition library
try:
    import speech_recognition
    print("SpeechRecognition library found.")
except ImportError:
    print("Warning: SpeechRecognition library not found. Install it: pip install SpeechRecognition")
    print("SpeechT5 Processor might fail without it.")

print("-------------------------------------------")
print("Hugging Face Local Inference Example")
print("Task: Text-to-Speech (TTS)")
print("Model: microsoft/speecht5_tts + microsoft/speecht5_hifigan")
print("-------------------------------------------")

# --- USER Configuration ---
# 1. Define the text you want to convert to speech
#    Using context relevant to Friday evening in Perth
text_to_speak = "It is Friday evening in Perth, time to relax after a long week, perhaps with a walk by the Swan River as the sun sets."

# 2. Choose a speaker embedding (optional - change index for different voice)
#    Index 7306 is a common example female voice from the dataset
speaker_index = 7306
output_filename = "tts_output.wav"
# --------------------------

# --- Load Speaker Embeddings ---
# Uses the datasets library to load x-vector speaker embeddings
print("\nLoading speaker embeddings dataset (may download on first run)...")
speaker_embedding = None
try:
    embeddings_dataset = datasets.load_dataset(
        "Matthijs/cmu-arctic-xvectors",
        split="validation",
        trust_remote_code=True # Required by this dataset
        )
    # Convert the selected embedding numpy array to a torch tensor
    speaker_embedding = torch.tensor(embeddings_dataset[speaker_index]["xvector"]).unsqueeze(0)
    # .unsqueeze(0) adds the batch dimension
    print(f"Speaker embedding loaded successfully (using index {speaker_index}).")
except ImportError:
    print("ERROR: 'datasets' library not found. Cannot load speaker embeddings.")
    print("Please install it: pip install datasets")
    exit()
except Exception as e:
    print(f"Error loading speaker embeddings dataset: {e}")
    print("Check internet connection or dataset availability.")
    exit()
# ------------------------------

# --- Model and Pipeline Loading ---
# Note: For TTS, the pipeline is often built manually from components for more control,
# but let's try the text-to-speech pipeline first if available and compatible.
# UPDATE: The standard pipeline often requires manual loading of components for SpeechT5.
# Let's load components manually.

print("\nLoading TTS model, Vocoder, and Processor (may download on first run)...")
synthesiser = None
processor = None
model = None
vocoder = None
try:
    # Determine device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda:0": print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load the processor (tokenizer)
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    # Load the main TTS model
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
    # Load the HiFi-GAN vocoder (converts spectrogram to waveform)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

    print("TTS components loaded successfully.")

except ImportError as e:
     print(f"Import Error: {e}. One of the required libraries might be missing.")
     print("Ensure transformers, torch, datasets, soundfile, SpeechRecognition, protobuf are installed.")
     exit()
except Exception as e:
    print(f"Error loading model components: {e}")
    exit()
# ---------------------------------

# --- Text-to-Speech Synthesis ---
print(f"\nSynthesizing speech for text: \"{text_to_speak}\"")
speech = None
sampling_rate = 16000 # SpeechT5 expects 16kHz
try:
    # 1. Process text input
    inputs = processor(text=text_to_speak, return_tensors="pt").to(device)

    # 2. Generate speech spectrogram
    #    Pass the speaker embeddings to the model
    spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings=speaker_embedding.to(device))

    # 3. Use Vocoder to generate waveform from spectrogram
    with torch.no_grad(): # Ensure no gradients are calculated for vocoder
        speech = vocoder(spectrogram).cpu().numpy() # Move to CPU and convert to numpy

    print("Speech synthesis complete.")

    # 4. Save the generated audio to a file
    print(f"Saving audio to {output_filename}...")
    # Ensure the array is 1D or suitable format for soundfile
    # SpeechT5 output might have a batch dim, remove it if present:
    if speech.ndim > 1 and speech.shape[0] == 1:
        speech_1d = speech.squeeze()
    else:
        speech_1d = speech

    sf.write(output_filename, speech_1d, samplerate=sampling_rate)
    print("Audio saved successfully.")
    print("\n--- Output ---")
    print(f"Generated speech saved to: {output_filename}")
    print("--------------")


except Exception as e:
    print(f"Error during speech synthesis or saving: {e}")

# --------------------------

print("\nExample finished.")