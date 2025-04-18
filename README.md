# Hugging Face Local Inference Examples

This repository contains a collection of Python scripts demonstrating how to run various AI tasks locally using models from the Hugging Face Hub and the `transformers` library (along with related libraries like `datasets` and `sentence-transformers`).

These examples cover a range of modalities including text, vision, and audio.

## Examples Included

This repository includes scripts for the following tasks (see list above or individual script headers for details):

1.  Sentiment Analysis
2.  Text Generation
3.  Zero-Shot Classification
4.  Named Entity Recognition (NER)
5.  Summarization
6.  Translation (EN->FR)
7.  Question Answering (Extractive)
8.  Fill-Mask (Masked Language Modeling)
9.  Sentence Embeddings & Similarity Search
10. Emotion Classification
11. Image Classification
12. Object Detection (with Annotation)
13. Automatic Speech Recognition (ASR)
14. Image Captioning

*(Refer to the list above or comments within each script for the specific models used).*

## Prerequisites

Before running these scripts, ensure you have the following:

1.  **Python:** Python 3.8 or later is recommended.
2.  **System Dependencies (Ubuntu/Debian):** Some scripts (especially audio-related) require system libraries.
    ```bash
    sudo apt update && sudo apt install libsndfile1 ffmpeg
    ```
    *(Other operating systems may require different commands to install equivalent libraries).*
3.  **Python Libraries:** It's highly recommended to use a Python virtual environment. You can install all common dependencies used across these examples with:
    ```bash
    pip install transformers torch datasets soundfile librosa sentence-transformers Pillow torchvision timm requests
    ```
    * **Note:** Not every script requires *all* of these libraries. Refer to individual script READMEs (if provided) or the `# Prerequisites` section within the script files for specific needs.

## General Usage

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Create Virtual Environment (Recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    *(Use `.\.venv\Scripts\activate` on Windows)*
3.  **Install System Dependencies:** Follow the instructions in the Prerequisites section if applicable for your OS.
4.  **Install Python Libraries:** Run the combined pip command from the Prerequisites section, or install libraries as needed for the specific script you want to run.
5.  **Configure Script Inputs (IMPORTANT):**
    * Many scripts require you to provide input, such as a path to a local image (`.jpg`, `.png`) or audio file (`.wav`, `.flac`).
    * **Open the specific `.py` script you want to run** in a text editor.
    * Look for a variable near the top, typically named `user_image_path`, `user_audio_path`, or similar.
    * **Modify the placeholder path** in that variable to point to your actual input file. Some scripts include logic to download a sample file if the specified path is not found. Read the comments in the script carefully.
6.  **Run the Script:**
    * Execute the desired script using Python from your terminal (ensure your virtual environment is active):
        ```bash
        python <script_name>.py
        ```
        (e.g., `python run_sentiment.py`, `python run_image_captioning.py`)

## Model Downloads

The first time you run a script using a specific Hugging Face model, the necessary model weights and configuration files will be automatically downloaded from the Hugging Face Hub and cached locally (usually in `~/.cache/huggingface/` or `C:\Users\<User>\.cache\huggingface\`). Subsequent runs using the same model will load directly from the cache, making them much faster and enabling offline use.

## Hardware Considerations

* **CPU:** Most scripts will run on a CPU, but performance (especially for larger models or complex tasks like vision/audio/generation) might be slow.
* **GPU:** An NVIDIA GPU with CUDA configured correctly and a compatible version of `torch` installed is highly recommended for significantly faster inference. The scripts include basic logic to attempt using the GPU if available.
* **RAM:** Models vary greatly in size. Ensure you have sufficient RAM. Smaller models might need 4-8GB, while larger ones (like `large` variants, vision models) might require 16GB or more.

## License

* The Python scripts in this repository are provided as examples, likely under the MIT License (or specify your chosen license).
* The Hugging Face libraries (`transformers`, `datasets`, etc.) are typically licensed under Apache 2.0.
* Individual models downloaded from the Hugging Face Hub have their own licenses. Please refer to the model card on the Hub for specific terms of use for each model.

---