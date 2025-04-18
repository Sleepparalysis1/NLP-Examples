# Local Image Captioning with ViT-GPT2

This script generates descriptive text captions for images locally using the `nlpconnect/vit-gpt2-image-captioning` model via the Hugging Face `transformers` library.

It is designed to be flexible:
1.  It will prioritize using a local image file path that you specify within the script.
2.  If the specified local file is not found, it will automatically download a sample image (surfers on a beach) and use that instead for demonstration purposes.

## Features

* Performs image captioning locally on your machine.
* Uses the `nlpconnect/vit-gpt2-image-captioning` model (ViT vision encoder + GPT-2 text decoder).
* Handles user-specified local image files.
* Includes a fallback to download a sample image if the local file is not found.
* Leverages the Hugging Face `transformers` library.
* Optionally utilizes GPU for faster processing if available and `torch` is installed with CUDA support.

## Model Used

* **Image Captioning Model:** `nlpconnect/vit-gpt2-image-captioning`

## Prerequisites

Before running the script, ensure you have the following installed:

1.  **Python:** Python 3.8 or later recommended.
2.  **System Dependencies:** None specific beyond standard build tools if installing libraries from source.
3.  **Python Libraries:** You can install these using pip. It is highly recommended to use a Python virtual environment.
    ```bash
    pip install transformers torch Pillow torchvision timm requests
    ```
    * `transformers`: The core Hugging Face library.
    * `torch`: The deep learning framework backend (PyTorch). Or install `tensorflow`.
    * `Pillow`: For loading and handling images.
    * `torchvision`, `timm`: Often required/beneficial for vision models and image preprocessing within `transformers`.
    * `requests`: Used to download the sample image if needed.

## Installation

1.  **Clone or Download:** Get the `run_image_captioning.py` script onto your local machine.
2.  **Create Virtual Environment (Recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    *(Use `.\.venv\Scripts\activate` on Windows)*
3.  **Install Python Libraries:** Run the pip command from the Prerequisites section within your activated virtual environment.

## Usage

1.  **Configure Image Input:**
    * Open the `run_image_captioning.py` script in a text editor.
    * Locate the line: `user_image_path = "my_caption_image.jpg"`
    * **Option A (Recommended):** Change the path `"my_caption_image.jpg"` to the *exact path* of the image file you want to caption on your computer (e.g., `/home/user/Pictures/perth_sunset.jpg`).
    * **Option B:** Place your image file in the *same directory* as the `run_image_captioning.py` script and ensure its name is exactly `my_caption_image.jpg`.
    * **Note:** If the script does not find a file at the specified `user_image_path`, it will attempt to download and use the sample image. Common image formats like JPG and PNG are supported.

2.  **Run the Script:**
    * Open your terminal or command prompt.
    * Make sure your virtual environment is activated (if you created one).
    * Navigate to the directory containing the script.
    * Execute the script using Python:
        ```bash
        python run_image_captioning.py
        ```

## Expected Output

The script will print status messages, including which image source is being used. The final output will be the generated caption(s) for the image: