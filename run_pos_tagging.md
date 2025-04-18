# Local Part-of-Speech (POS) Tagging with BERT

This script performs Part-of-Speech (POS) tagging locally using the `vblagoje/bert-english-uncased-finetuned-pos` model via the Hugging Face `transformers` library's `token-classification` pipeline.

It takes an input text sentence and identifies the grammatical role (Noun, Verb, Adjective, Preposition, etc.) of each word or token based on the Penn Treebank (PTB) tag set.

## Features

* Performs POS tagging locally on your machine.
* Uses a BERT-based model (`vblagoje/bert-english-uncased-finetuned-pos`) fine-tuned for English POS tagging.
* Identifies the part-of-speech tag for each word/token.
* Leverages the Hugging Face `transformers` library (`token-classification` pipeline).
* Optionally utilizes GPU for faster processing.

## Model Used

* **POS Tagging Model:** `vblagoje/bert-english-uncased-finetuned-pos`

## Prerequisites

Before running the script, ensure you have the following installed:

1.  **Python:** Python 3.8 or later recommended.
2.  **System Dependencies:** None specific beyond standard build tools.
3.  **Python Libraries:** Install using pip in a virtual environment. Only core libraries are needed for this text-based task.
    ```bash
    pip install transformers torch
    ```
    * `transformers`: The core Hugging Face library.
    * `torch`: The deep learning framework backend (PyTorch).

## Installation

1.  **Clone or Download:** Get the `run_pos_tagging.py` script onto your local machine.
2.  **Create Virtual Environment (Recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    *(Use `.\.venv\Scripts\activate` on Windows)*
3.  **Install Python Libraries:** Run the pip command from the Prerequisites section within your activated virtual environment.

## Usage

1.  **Configure Text Input (Optional):**
    * Open the `run_pos_tagging.py` script in a text editor.
    * Locate the line: `text_to_tag = "..."`
    * Modify the text string inside the quotes to the sentence you want to tag.

2.  **Run the Script:**
    * Open your terminal or command prompt.
    * Make sure your virtual environment is activated.
    * Navigate to the directory containing the script.
    * Execute the script using Python:
        ```bash
        python run_pos_tagging.py
        ```

## Expected Output

The script will print the input text and then list each recognized word/token along with its predicted Part-of-Speech tag (based on the Penn Treebank tag set) and confidence score.