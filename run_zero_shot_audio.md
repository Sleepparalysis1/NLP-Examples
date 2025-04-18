# Local Zero-Shot Audio Classification with CLAP

This script performs zero-shot audio classification locally using the `laion/clap-htsat-unfused` model via the Hugging Face `transformers` library.

You provide an audio file and a list of candidate text labels (describing sounds), and the script predicts how well the audio matches each label, even if the model wasn't explicitly trained on those exact sound categories. This leverages the CLAP model's ability to understand relationships between audio and text descriptions.

This version explicitly loads the audio into a NumPy array, converts it to mono, and resamples it to the expected sample rate (48kHz for this CLAP model) before passing it to the pipeline.

It includes flexibility for the audio input:
1.  It prioritizes using a local audio file path specified within the script.
2.  If the specified file isn't found or fails to load, it falls back to using a sample speech audio clip (from `librispeech_asr_dummy`), which is also preprocessed.

## Features

* Performs zero-shot audio classification locally.
* Uses the `laion/clap-htsat-unfused` model (CLAP architecture).
* Classifies audio against user-provided text labels at runtime.
* Handles user-specified local audio files with a fallback to a sample audio clip.
* Includes audio preprocessing (Mono conversion, Resampling to 48kHz).
* Leverages the Hugging Face `transformers` and `datasets` libraries.
* Optionally utilizes GPU for faster processing.

## Model Used

* **Zero-Shot Audio Classification Model:** `laion/clap-htsat-unfused`

## Prerequisites

Before running the script, ensure you have the following installed:

1.  **Python:** Python 3.8 or later recommended.
2.  **System Dependencies (Ubuntu/Debian):** Essential for audio file handling by underlying libraries.
    ```bash
    sudo apt update && sudo apt install libsndfile1 ffmpeg
    ```
    *(Other operating systems may require different commands).*
3.  **Python Libraries:** Install using pip in a virtual environment. Includes libraries for core functionality, audio handling, array manipulation, and dataset loading.
    ```bash
    pip install "transformers[audio]" torch datasets soundfile librosa numpy
    ```
    * `transformers[audio]`: Core library with extras for audio (like `librosa`).
    * `torch`: Deep learning framework backend.
    * `datasets`: Used to download the sample audio file if needed.
    * `soundfile`: Used for reading audio files.
    * `librosa`: Used for audio analysis, required for resampling.
    * `numpy`: For numerical array manipulation.

## Installation

1.  **Clone or Download:** Get the `run_zero_shot_audio.py` script onto your local machine.
2.  **Create Virtual Environment (Recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    *(Use `.\.venv\Scripts\activate` on Windows)*
3.  **Install System Dependencies:** Follow instructions in Prerequisites.
4.  **Install Python Libraries:** Run the pip command from Prerequisites within your activated virtual environment.

## Usage

1.  **Configure Inputs (Audio & Labels):**
    * Open the `run_zero_shot_audio.py` script in a text editor.
    * **Audio:**
        * Locate the line: `user_audio_path = "my_zero_shot_audio.wav"`
        * **Option A (Recommended):** Change the path to the *exact path* of the audio file you want to classify.
        * **Option B:** Place your audio file in the *same directory* as the script and name it `my_zero_shot_audio.wav`.
        * **Fallback:** If no file is found or loadable at `user_audio_path`, the script uses the sample speech audio.
    * **Candidate Labels:**
        * Locate the list: `candidate_labels = [...]`
        * **Modify this list** to include text descriptions of sounds you want to classify the audio against.

2.  **Run the Script:**
    * Open your terminal or command prompt.
    * Make sure your virtual environment is activated.
    * Navigate to the directory containing the script.
    * Execute the script using Python:
        ```bash
        python run_zero_shot_audio.py
        ```

## Expected Output

The script will print status messages, including the audio source used and confirmation of preprocessing steps (like resampling). The final output will be a list of the candidate text labels ranked by their predicted relevance score to the audio content: