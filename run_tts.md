# Local Text-to-Speech (TTS) with SpeechT5

This script performs Text-to-Speech (TTS) synthesis locally using the `microsoft/speecht5_tts` model and the `microsoft/speecht5_hifigan` vocoder via the Hugging Face `transformers` library.

It takes a text string as input, uses pre-defined speaker characteristics (embeddings loaded from the `Matthijs/cmu-arctic-xvectors` dataset), generates the corresponding speech audio waveform, and saves it as a WAV file.

## Features

* Performs text-to-speech synthesis locally on your machine.
* Uses the `microsoft/speecht5_tts` model for text-to-spectrogram conversion.
* Uses the `microsoft/speecht5_hifigan` vocoder for high-quality waveform generation from the spectrogram.
* Utilizes pre-computed speaker embeddings for specific voice characteristics (loaded via `datasets` library).
* Saves the generated speech as a standard WAV audio file (`tts_output.wav`).
* Leverages the Hugging Face `transformers` and `datasets` libraries.
* Optionally utilizes GPU for faster processing.

## Models Used

* **TTS Model:** `microsoft/speecht5_tts`
* **Vocoder Model:** `microsoft/speecht5_hifigan`
* **Speaker Embeddings:** From `Matthijs/cmu-arctic-xvectors` dataset on Hugging Face Hub.

## Prerequisites

Before running the script, ensure you have the following installed:

1.  **Python:** Python 3.8 or later recommended.
2.  **System Dependencies (Ubuntu/Debian):** `libsndfile` is needed for saving audio files. `ffmpeg` is generally recommended for broader audio library compatibility.
    ```bash
    sudo apt update && sudo apt install libsndfile1 ffmpeg
    ```
    *(Other operating systems may require different commands).*
3.  **Python Libraries:** Install using pip in a virtual environment. SpeechT5 requires specific dependencies.
    ```bash
    pip install transformers torch datasets soundfile SpeechRecognition protobuf
    ```
    * `transformers`: The core Hugging Face library.
    * `torch`: The deep learning framework backend (PyTorch).
    * `datasets`: Used to download the speaker embeddings dataset.
    * `soundfile`: Required for saving the output WAV file.
    * `SpeechRecognition`: Often required by the SpeechT5 processor/tokenizer.
    * `protobuf`: Often a dependency for certain model operations.

## Installation

1.  **Clone or Download:** Get the `run_tts.py` script onto your local machine.
2.  **Create Virtual Environment (Recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    *(Use `.\.venv\Scripts\activate` on Windows)*
3.  **Install System Dependencies:** Follow instructions in Prerequisites.
4.  **Install Python Libraries:** Run the pip command from Prerequisites within your activated virtual environment.

## Usage

1.  **Configure Text Input (Optional):**
    * Open the `run_tts.py` script in a text editor.
    * Locate the line: `text_to_speak = "..."`
    * Modify the text string inside the quotes to the text you want synthesized.
    * (Advanced) You can change the `speaker_index` variable (default is `7306`) to select a different voice from the available speaker embeddings in the dataset.

2.  **Run the Script:**
    * Open your terminal or command prompt.
    * Make sure your virtual environment is activated.
    * Navigate to the directory containing the script.
    * Execute the script using Python:
        ```bash
        python run_tts.py
        ```

## Expected Output

The script will print status messages, including confirmation of model loading and which speaker embedding index is used. The primary output is not text printed to the console, but an audio file:
* A **speech audio file** will be saved as **`tts_output.wav`** in the same directory where you run the script.
* You can play this WAV file using any standard audio player to hear the synthesized speech of the input text using the chosen speaker's voice characteristics.

## Troubleshooting

* **Library/System Errors:** Ensure `libsndfile1`, `ffmpeg` (recommended), and all required Python libraries (`transformers`, `torch`, `datasets`, `soundfile`, `SpeechRecognition`, `protobuf`) are correctly installed in the active virtual environment.
* **Model/Dataset Download Issues:** Check your internet connection. The TTS model, vocoder, and speaker embedding dataset need to be downloaded on the first run and can be large.
* **Audio Quality:** The quality depends on the models and speaker embedding. SpeechT5 generally produces good quality speech. Ensure your system's audio playback is working correctly.
* **Errors during Synthesis:** Check the console for specific errors. Ensure the input text doesn't contain highly unusual characters that might cause issues for the processor. Check available RAM/GPU memory.

## Hardware Considerations

* **CPU:** Possible, but TTS synthesis (especially the vocoder step generating the waveform) can be computationally intensive and quite slow on a CPU.
* **GPU:** An NVIDIA GPU is highly recommended for generating speech in a reasonable amount of time.
* **RAM:** Ensure sufficient RAM for loading the TTS model, vocoder, embeddings, and handling the generated audio waveform.

## License

* The `run_tts.py` script itself is provided as an example (consider MIT License).
* Hugging Face libraries and `datasets` are typically Apache 2.0 licensed. `soundfile` uses BSD/MIT-like licenses. `SpeechRecognition` uses a BSD license. `protobuf` uses a BSD-style license.
* The SpeechT5 models (`microsoft/speecht5_tts`, `microsoft/speecht5_hifigan`) are available under the MIT license. The speaker embedding dataset (`Matthijs/cmu-arctic-xvectors`) should be checked for its specific terms (likely permissive for research/non-commercial use, but always verify on the dataset card).