# Local Visual Question Answering (VQA) with ViLT

This script performs Visual Question Answering (VQA) locally. You provide an image and a question about its content, and the script uses the `dandelin/vilt-b32-finetuned-vqa` model via the Hugging Face `transformers` library to generate an answer based on the image's visual content.

It includes flexibility for the image input:
1.  It prioritizes using a local image file path specified within the script.
2.  If the specified file isn't found, it downloads a sample image (a COCO dataset image featuring cats and a remote) for demonstration.

## Features

* Performs VQA locally on your machine.
* Uses the `dandelin/vilt-b32-finetuned-vqa` model (ViLT: Vision-and-Language Transformer).
* Answers natural language questions based on image content.
* Handles user-specified local image files with a fallback to a sample image.
* Leverages the Hugging Face `transformers` library.
* Optionally utilizes GPU for faster processing.

## Model Used

* **VQA Model:** `dandelin/vilt-b32-finetuned-vqa`

## Prerequisites

Before running the script, ensure you have the following installed:

1.  **Python:** Python 3.8 or later recommended.
2.  **System Dependencies:** None specific beyond standard build tools if installing libraries from source.
3.  **Python Libraries:** You can install these using pip. Using a Python virtual environment is highly recommended.
    ```bash
    pip install transformers torch Pillow torchvision timm requests
    ```
    * `transformers`: The core Hugging Face library.
    * `torch`: The deep learning framework backend (PyTorch).
    * `Pillow`: For loading and handling images.
    * `torchvision`, `timm`: Often required/beneficial for vision models and image preprocessing.
    * `requests`: Used to download the sample image if needed.

## Installation

1.  **Clone or Download:** Get the `run_vqa.py` script onto your local machine.
2.  **Create Virtual Environment (Recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    *(Use `.\.venv\Scripts\activate` on Windows)*
3.  **Install Python Libraries:** Run the pip command from the Prerequisites section within your activated virtual environment.

## Usage

1.  **Configure Inputs (Image & Question):**
    * Open the `run_vqa.py` script in a text editor.
    * **Image:**
        * Locate the line: `user_image_path = "my_vqa_image.jpg"`
        * **Option A (Recommended):** Change the path `"my_vqa_image.jpg"` to the *exact path* of the image file you want to ask questions about.
        * **Option B:** Place your image file in the *same directory* as the script and name it `my_vqa_image.jpg`.
        * **Fallback:** If no file is found at `user_image_path`, the script downloads and uses a sample image (`vqa_sample_image.jpg`).
    * **Question:**
        * Locate the line: `question = "How many cats are in the picture?"`
        * **Change the question text** to be relevant to the image you are providing (or the sample image).

2.  **Run the Script:**
    * Open your terminal or command prompt.
    * Make sure your virtual environment is activated.
    * Navigate to the directory containing the script.
    * Execute the script using Python:
        ```bash
        python run_vqa.py
        ```

## Expected Output

The script will print status messages, including the image source used and the question being asked. The final output will be the model's answer(s) based on the image: