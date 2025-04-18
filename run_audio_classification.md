# Local Audio Classification with AST

This script performs audio classification locally using the `MIT/ast-finetuned-audioset-10-10-0.4593` model via the Hugging Face `transformers` library. It predicts the type of sound present in an audio file (e.g., Speech, Music, Dog, Siren, etc., based on the AudioSet categories).

It supports using a local audio file or falling back to a sample:
1.  It prioritizes using a local audio file path specified within the script.
2.  If the specified file isn't found, it uses the same sample audio clip (from `librispeech_asr_dummy`, which contains speech) as the ASR example.

## Features

* Performs audio classification locally on your machine.
* Uses the `MIT/ast-finetuned-audioset-10-10-0.4593` model (Audio Spectrogram Transformer).
* Classifies audio into general sound categories (based on AudioSet).
* Handles user-specified local audio files with a fallback to a sample audio clip.
* Leverages the Hugging Face `transformers` and `datasets` libraries.
* Optionally utilizes GPU for faster processing.

## Model Used

* **Audio Classification Model:** `MIT/ast-finetuned-audioset-10-10-0.4593`

## Prerequisites

Before running the script, ensure you have the following installed:

1.  **Python:** Python 3.8 or later recommended.
2.  **System Dependencies (Ubuntu/Debian):** Audio processing often relies on these. `ffmpeg` is recommended for broader format support by underlying libraries.
    ```bash
    sudo apt update && sudo apt install libsndfile1 ffmpeg
    ```
    *(Other operating systems may require different commands).*
3.  **Python Libraries:** Install using pip in a virtual environment. `torchaudio` is often needed for AST models' feature extraction.
    ```bash
    pip install transformers torch datasets soundfile librosa torchaudio
    ```
    * `transformers`: The core Hugging Face library.
    * `torch`: The deep learning framework backend (PyTorch).
    * `datasets`: Used to download the sample audio file if needed.
    * `soundfile`, `librosa`: Common libraries for audio I/O and processing.
    * `torchaudio`: PyTorch's audio library, often used for feature extraction (like spectrograms) for audio models.

## Installation

1.  **Clone or Download:** Get the `run_audio_classification.py` script onto your local machine.
2.  **Create Virtual Environment (Recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    *(Use `.\.venv\Scripts\activate` on Windows)*
3.  **Install System Dependencies:** Follow instructions in Prerequisites.
4.  **Install Python Libraries:** Run the pip command from Prerequisites within your activated virtual environment.

## Usage

1.  **Configure Audio Input:**
    * Open the `run_audio_classification.py` script in a text editor.
    * Locate the line: `user_audio_path = "my_audio_for_classification.wav"`
    * **Option A (Recommended):** Change the path to the *exact path* of the audio file you want to classify (e.g., `.wav`, `.flac`, `.mp3`). Try using sounds like speech, music, dog barking, etc.
    * **Option B:** Place your audio file in the *same directory* as the script and name it `my_audio_for_classification.wav`.
    * **Fallback:** If no file is found at `user_audio_path`, the script uses the sample speech audio from `librispeech_asr_dummy`.

2.  **Run the Script:**
    * Open your terminal or command prompt.
    * Make sure your virtual environment is activated.
    * Navigate to the directory containing the script.
    * Execute the script using Python:
        ```bash
        python run_audio_classification.py
        ```

## Expected Output

The script will print status messages, including the audio source used. The final output will be a list of the top predicted sound categories for the audio file.