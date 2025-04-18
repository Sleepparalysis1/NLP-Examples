# Local Automatic Speech Recognition (ASR) with Whisper

This script performs automatic speech recognition (speech-to-text) locally using the `openai/whisper-base` model via the Hugging Face `transformers` library.

It is designed to be flexible:
1.  It will prioritize using a local audio file path that you specify within the script.
2.  If the specified local file is not found, it will automatically download a short sample audio clip from the Hugging Face Hub (`hf-internal-testing/librispeech_asr_dummy`) and use that instead for demonstration purposes.

## Features

* Performs ASR locally on your machine.
* Uses the efficient `openai/whisper-base` model.
* Prioritizes user-specified local audio file.
* Includes a fallback to download a sample audio file if the local file is not found.
* Leverages the Hugging Face `transformers` and `datasets` libraries.
* Optionally utilizes GPU for faster processing if available and `torch` is installed with CUDA support.

## Model Used

* **ASR Model:** `openai/whisper-base`

## Prerequisites

Before running the script, ensure you have the following installed:

1.  **Python:** Python 3.8 or later recommended.
2.  **System Dependencies (Ubuntu/Debian):**
    * `libsndfile`: Required by the `soundfile` Python library for reading/writing audio files.
    * `ffmpeg`: Required by underlying libraries (like `librosa` or `transformers`) for decoding various audio formats when loading from a filename.
    ```bash
    sudo apt update && sudo apt install libsndfile1 ffmpeg
    ```
3.  **Python Libraries:** You can install these using pip. It is highly recommended to use a Python virtual environment.
    ```bash
    pip install transformers torch datasets soundfile librosa
    ```
    * `transformers`: The core Hugging Face library.
    * `torch`: The deep learning framework backend (PyTorch). Or install `tensorflow`.
    * `datasets`: Used to download the sample audio file if your local file isn't found.
    * `soundfile`: Used for handling audio file operations.
    * `librosa`: Provides advanced audio analysis features, often required by `datasets` or `transformers` for audio loading/processing.

## Installation

1.  **Clone or Download:** Get the `run_asr_flexible.py` script onto your local machine.
2.  **Create Virtual Environment (Recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    *(Use `.\.venv\Scripts\activate` on Windows)*
3.  **Install System Dependencies:** Follow the instructions in the Prerequisites section above.
4.  **Install Python Libraries:** Run the pip command from the Prerequisites section within your activated virtual environment.

## Usage

1.  **Configure Audio Input:**
    * Open the `run_asr_flexible.py` script in a text editor.
    * Locate the line: `user_audio_path = "my_audio.wav"`
    * **Option A (Recommended):** Change the path `"my_audio.wav"` to the *exact path* of the audio file you want to transcribe on your computer (e.g., `/home/user/recordings/meeting.wav` or `C:/Users/user/Documents/sound.mp3`).
    * **Option B:** Place your audio file in the *same directory* as the `run_asr_flexible.py` script and ensure its name is exactly `my_audio.wav`.
    * **Note:** If the script does not find a file at the specified `user_audio_path`, it will attempt to download and use the sample audio. Common audio formats like WAV and FLAC are generally well-supported. Other formats like MP3 often depend on the `ffmpeg` installation.

2.  **Run the Script:**
    * Open your terminal or command prompt.
    * Make sure your virtual environment is activated (if you created one).
    * Navigate to the directory containing the script.
    * Execute the script using Python:
        ```bash
        python run_asr_flexible.py
        ```

## Expected Output

The script will print status messages to the console, including:
* Which audio source is being used (your local file or the downloaded sample).
* Confirmation of model loading and the device being used (CPU or GPU).
* The final transcription result, like:

    ```
    --- Transcription Result ---
    Recognized Text: "The birch canoe slid on the smooth planks."
    ----------------------------
    ```
    *(The exact text will depend on the audio input)*

## Troubleshooting

* **`libsndfile` errors:** Ensure you ran `sudo apt install libsndfile1`.
* **`ffmpeg was not found`:** Ensure you ran `sudo apt install ffmpeg`.
* **`datasets` library not found:** Run `pip install datasets` (only needed for the fallback sample).
* **`soundfile`/`librosa` not found:** Run `pip install soundfile librosa`.
* **Errors during transcription:** Ensure your audio file is not corrupted and is in a reasonably common format. Check console for specific error messages. Very long audio files might require adjusting pipeline parameters (not implemented in this basic script).

## Hardware Considerations

* **CPU:** The script will run on a CPU, but transcription speed will depend on your processor and the audio length.
* **GPU:** If you have an NVIDIA GPU with CUDA set up correctly and the appropriate version of `torch` installed, the script will automatically use it for significantly faster processing.
* **RAM:** Model loading and processing require a moderate amount of RAM (a few GB should be sufficient for the `whisper-base` model).

## License

* The `run_asr_flexible.py` script itself is provided as an example (consider adding an MIT License if distributing).
* The Hugging Face libraries (`transformers`, `datasets`) are typically under the Apache 2.0 License.
* The `openai/whisper-base` model has its own license terms (generally permissive for research/use, but check the model card on Hugging Face Hub for specifics).