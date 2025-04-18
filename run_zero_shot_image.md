# Local Zero-Shot Image Classification with CLIP

This script performs zero-shot image classification locally using the `openai/clip-vit-base-patch32` model via the Hugging Face `transformers` library.

You provide an image and a list of candidate text labels, and the script predicts how well the image matches each label, even if the model wasn't explicitly trained on those specific labels beforehand. This is achieved using CLIP (Contrastive Language-Image Pre-training), which learns joint representations of images and text.

It includes flexibility for the image input:
1.  It prioritizes using a local image file path specified within the script.
2.  If the specified file isn't found, it downloads a sample image (COCO image with cats/remote) for demonstration.

## Features

* Performs zero-shot image classification locally.
* Uses the `openai/clip-vit-base-patch32` model (CLIP architecture: ViT + Text Transformer).
* Classifies images against user-provided text labels at runtime.
* Handles user-specified local image files with a fallback to a sample image.
* Leverages the Hugging Face `transformers` library.
* Optionally utilizes GPU for faster processing.

## Model Used

* **Zero-Shot Image Classification Model:** `openai/clip-vit-base-patch32`

## Prerequisites

Before running the script, ensure you have the following installed:

1.  **Python:** Python 3.8 or later recommended.
2.  **System Dependencies:** None specific beyond standard build tools.
3.  **Python Libraries:** Install using pip in a virtual environment. CLIP models often require `ftfy` and `regex` for text processing.
    ```bash
    pip install transformers torch Pillow torchvision timm requests ftfy regex
    ```
    * `transformers`: The core Hugging Face library.
    * `torch`: The deep learning framework backend (PyTorch).
    * `Pillow`: For loading and handling images.
    * `torchvision`, `timm`: Often required/beneficial for vision models.
    * `requests`: Used to download the sample image if needed.
    * `ftfy`: Fixes unicode mistakes, often used in CLIP's text tokenizer.
    * `regex`: Advanced regular expression library, sometimes needed by tokenizers.

## Installation

1.  **Clone or Download:** Get the `run_zero_shot_image.py` script onto your local machine.
2.  **Create Virtual Environment (Recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    *(Use `.\.venv\Scripts\activate` on Windows)*
3.  **Install Python Libraries:** Run the pip command from the Prerequisites section within your activated virtual environment.

## Usage

1.  **Configure Inputs (Image & Labels):**
    * Open the `run_zero_shot_image.py` script in a text editor.
    * **Image:**
        * Locate the line: `user_image_path = "my_zero_shot_image.jpg"`
        * **Option A (Recommended):** Change the path `"my_zero_shot_image.jpg"` to the *exact path* of the image file you want to classify.
        * **Option B:** Place your image file in the *same directory* as the script and name it `my_zero_shot_image.jpg`.
        * **Fallback:** If no file is found at `user_image_path`, the script downloads and uses the sample image (`zero_shot_sample_image.jpg`).
    * **Candidate Labels:**
        * Locate the list: `candidate_labels = [...]`
        * **Modify this list** to include the text labels you want to classify the image against. Make them relevant to your image (or the sample image).

2.  **Run the Script:**
    * Open your terminal or command prompt.
    * Make sure your virtual environment is activated.
    * Navigate to the directory containing the script.
    * Execute the script using Python:
        ```bash
        python run_zero_shot_image.py
        ```

## Expected Output

The script will print status messages, the image source, and the candidate labels used. The final output will be a list of the candidate labels ranked by their predicted relevance (similarity score) to the image content: