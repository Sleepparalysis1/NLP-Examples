# Local Document Question Answering (DocVQA) with Donut

This script performs Document Question Answering (DocVQA) locally using the `naver-clova-ix/donut-base-finetuned-docvqa` model via the Hugging Face `transformers` library.

You provide an image of a document (e.g., receipt, invoice, form) and a question about its content. The Donut model analyzes the image, reading text and understanding layout **without requiring a separate OCR step**, to find the answer within the document image.

It includes flexibility for the image input:
1.  It prioritizes using a local document image file path specified within the script.
2.  If the specified file isn't found, it downloads a sample image (a simple receipt) for demonstration.

## Features

* Performs Document Question Answering locally.
* Uses the `naver-clova-ix/donut-base-finetuned-docvqa` model (Donut architecture, OCR-free).
* Answers natural language questions based on document image content and layout.
* Handles user-specified local document image files with a fallback to a sample image.
* Leverages the Hugging Face `transformers` library.
* Optionally utilizes GPU for faster processing.

## Model Used

* **DocVQA Model:** `naver-clova-ix/donut-base-finetuned-docvqa`

## Prerequisites

Before running the script, ensure you have the following installed:

1.  **Python:** Python 3.8 or later recommended.
2.  **System Dependencies:** None specific beyond standard build tools.
3.  **Python Libraries:** Install using pip in a virtual environment. Donut requires `sentencepiece`. Standard vision libraries are also recommended.
    ```bash
    pip install transformers torch Pillow torchvision timm requests sentencepiece
    ```
    * `transformers`: The core Hugging Face library.
    * `torch`: The deep learning framework backend (PyTorch).
    * `Pillow`: For loading and handling images.
    * `torchvision`, `timm`: Often required/beneficial for underlying vision components.
    * `requests`: Used to download the sample document image if needed.
    * `sentencepiece`: Required by the Donut model's tokenizer.

## Installation

1.  **Clone or Download:** Get the `run_docvqa.py` script onto your local machine.
2.  **Create Virtual Environment (Recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    *(Use `.\.venv\Scripts\activate` on Windows)*
3.  **Install Python Libraries:** Run the pip command from the Prerequisites section within your activated virtual environment.

## Usage

1.  **Configure Inputs (Document Image & Question):**
    * Open the `run_docvqa.py` script in a text editor.
    * **Document Image:**
        * Locate the line: `user_doc_image_path = "my_document.png"`
        * **Option A (Recommended):** Change the path to the *exact path* of the document image file (e.g., receipt.jpg, form.png) you want to query.
        * **Option B:** Place your document image file in the *same directory* as the script and name it `my_document.png` (or `.jpg` etc., and update the variable).
        * **Fallback:** If no file is found at `user_doc_image_path`, the script downloads and uses the sample receipt image (`docvqa_sample_receipt.jpg`).
    * **Question:**
        * Locate the line: `question = "What is the total amount?"`
        * **Change the question text** to be relevant to the content of the document image you are providing (or the sample receipt).

2.  **Run the Script:**
    * Open your terminal or command prompt.
    * Make sure your virtual environment is activated.
    * Navigate to the directory containing the script.
    * Execute the script using Python:
        ```bash
        python run_docvqa.py
        ```

## Expected Output

The script will print status messages, the image source used, and the question asked. The final output will be the model's answer extracted from the document image: