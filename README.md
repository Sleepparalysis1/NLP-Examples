# Hugging Face Local Inference Examples

This repository contains a collection of Python scripts demonstrating how to run various AI tasks locally using models from the Hugging Face Hub and the `transformers` library (along with related libraries like `datasets`, `sentence-transformers`, etc.).

These examples cover a range of modalities including text, vision, and audio, showcasing different models and pipelines available within the Hugging Face ecosystem.

## Examples Included

This repository includes scripts for the following tasks (see list below or individual script headers for details):

1.  **Sentiment Analysis (`run_sentiment.py`)**
    * Task: Text Classification (Positive/Negative)
    * Model: `distilbert-base-uncased-finetuned-sst-2-english` (Pipeline Default)
2.  **Text Generation (`run_generation.py`)**
    * Task: Generating text following a prompt.
    * Model: `gpt2`
3.  **Zero-Shot Classification (`run_zero_shot.py`)**
    * Task: Classifying text using arbitrary labels without specific fine-tuning.
    * Model: `facebook/bart-large-mnli` (Pipeline Default)
4.  **Named Entity Recognition (NER) (`run_ner.py`)**
    * Task: Identifying named entities (Person, Location, Org).
    * Model: `dbmdz/bert-large-cased-finetuned-conll03-english`
5.  **Summarization (`run_summarization.py`)**
    * Task: Creating a shorter summary of a longer text.
    * Model: `facebook/bart-large-cnn`
6.  **Translation (EN->FR) (`run_translation.py`)**
    * Task: Translating text from English to French.
    * Model: `Helsinki-NLP/opus-mt-en-fr`
7.  **Question Answering (Extractive) (`run_qa.py`)**
    * Task: Finding the answer span within a context paragraph given a question.
    * Model: `distilbert-base-cased-distilled-squad`
8.  **Fill-Mask (`run_fill_mask.py`)**
    * Task: Predicting masked words in a sentence (Masked Language Modeling).
    * Model: `roberta-base`
9.  **Sentence Embeddings & Similarity (`run_embeddings.py`, `run_similarity_search.py`)**
    * Task: Generating semantic vector representations and finding similar sentences.
    * Model: `sentence-transformers/all-MiniLM-L6-v2` (via `sentence-transformers` library)
10. **Emotion Classification (`run_emotion.py`)**
    * Task: Text Classification (Detecting emotions like joy, anger, sadness).
    * Model: `j-hartmann/emotion-english-distilroberta-base`
11. **Image Classification (`run_image_classification.py`)**
    * Task: Classifying the main subject of an image.
    * Model: `google/vit-base-patch16-224`
12. **Object Detection (`run_object_detection_annotated.py`)**
    * Task: Identifying multiple objects in an image with bounding boxes and labels (plus annotation).
    * Model: `facebook/detr-resnet-50`
13. **Automatic Speech Recognition (ASR) (`run_asr_flexible.py`)**
    * Task: Transcribing speech from an audio file to text.
    * Model: `openai/whisper-base`
14. **Image Captioning (`run_image_captioning.py`)**
    * Task: Generating a text description for an image.
    * Model: `nlpconnect/vit-gpt2-image-captioning`
15. **Visual Question Answering (VQA) (`run_vqa.py`)**
    * Task: Answering questions based on image content.
    * Model: `dandelin/vilt-b32-finetuned-vqa`
16. **Table Question Answering (`run_table_qa.py`)**
    * Task: Answering questions based on tabular data.
    * Model: `google/tapas-base-finetuned-wtq`
17. **Audio Classification (`run_audio_classification.py`)**
    * Task: Classifying the type of sound in an audio file.
    * Model: `MIT/ast-finetuned-audioset-10-10-0.4593`
18. **Zero-Shot Image Classification (`run_zero_shot_image.py`)**
    * Task: Classifying images against arbitrary text labels.
    * Model: `openai/clip-vit-base-patch32`

*(Refer to comments within each script for more specific details on models and implementation.)*

## Prerequisites

Before running these scripts, ensure you have the following:

1.  **Python:** Python 3.8 or later is recommended.
2.  **System Dependencies (Ubuntu/Debian):** Some scripts (especially audio-related) require system libraries. Install common ones using:
    ```bash
    sudo apt update && sudo apt install libsndfile1 ffmpeg
    ```
    *(Other operating systems may require different commands to install equivalent libraries).*
3.  **Python Libraries:** It's highly recommended to use a Python virtual environment. You can install all common dependencies used across these examples with:
    ```bash
    pip install transformers torch datasets soundfile librosa sentence-transformers Pillow torchvision timm requests pandas torch-scatter ftfy regex torchaudio
    ```
    * **Note:** Not every script requires *all* of these libraries. However, installing them all ensures you can run any example. Refer to individual script READMEs (if provided) or comments within the files for minimal requirements.

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
4.  **Install Python Libraries:** Run the combined pip command from the Prerequisites section within your activated virtual environment.
5.  **Configure Script Inputs (IMPORTANT):**
    * Many scripts require you to provide input, such as a path to a local **image file** (`.jpg`, `.png`), an **audio file** (`.wav`, `.flac`), specific **text/questions**, **candidate labels**, or **table data** inside the script.
    * **Open the specific `.py` script you want to run** in a text editor before executing it.
    * Look for comments indicating `USER ACTION REQUIRED` or variables like `user_image_path`, `user_audio_path`, `question`, `candidate_labels`, `data` (for tables), etc.
    * **Modify these variables** according to the script's needs (e.g., provide a valid file path, change the question text, update labels, define table data). Some scripts include logic to download a sample file if a local one isn't found - read the script comments for details.
6.  **Run the Script:**
    * Execute the desired script using Python from your terminal (ensure your virtual environment is active):
        ```bash
        python <script_name>.py
        ```
        (e.g., `python run_sentiment.py`, `python run_zero_shot_image.py`)

## Model Downloads

The first time you run a script using a specific Hugging Face model, the necessary model weights, configuration, and tokenizer/processor files will be automatically downloaded from the Hugging Face Hub and cached locally (usually in `~/.cache/huggingface/` or `C:\Users\<User>\.cache\huggingface\`). Subsequent runs using the same model will load directly from the cache, making them much faster and enabling offline use (provided all necessary files are cached).

## Hardware Considerations

* **CPU:** Most scripts will run on a CPU, but performance (especially for larger models or complex tasks like vision, audio, generation) might be slow.
* **GPU:** An NVIDIA GPU with CUDA configured correctly and a compatible version of `torch` installed is highly recommended for significantly faster inference. The scripts include basic logic to attempt using the GPU if available.
* **RAM:** Models vary greatly in size. Ensure you have sufficient RAM. Smaller models might need 4-8GB, while larger ones (like `large` variants, vision/audio models) might require 16GB or more.

## License

* The Python scripts in this repository are provided as examples, likely under the MIT License (or specify your chosen license).
* The Hugging Face libraries (`transformers`, `datasets`, etc.) are typically licensed under Apache 2.0.
* Individual models downloaded from the Hugging Face Hub have their own licenses. Please refer to the model card on the Hub for specific terms of use for each model.