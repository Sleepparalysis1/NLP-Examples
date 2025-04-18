# Local Table Question Answering with TAPAS

This script performs Table Question Answering (Table QA) locally using the `google/tapas-base-finetuned-wtq` model via the Hugging Face `transformers` library.

You provide a table (as a Pandas DataFrame within the script) and a question about the data in the table, and the model will attempt to find the answer within that table structure.

## Features

* Performs Table QA locally on your machine.
* Uses the `google/tapas-base-finetuned-wtq` model (TAPAS architecture).
* Answers natural language questions based on structured table data.
* Requires table data defined as a Pandas DataFrame in the script.
* Leverages the Hugging Face `transformers` and `pandas` libraries.
* Optionally utilizes GPU for faster processing.

## Model Used

* **Table QA Model:** `google/tapas-base-finetuned-wtq` (fine-tuned on WikiTableQuestions)

## Prerequisites

Before running the script, ensure you have the following installed:

1.  **Python:** Python 3.8 or later recommended.
2.  **System Dependencies:** None specific beyond standard build tools.
3.  **Python Libraries:** You need `pandas` for table handling and potentially `torch-scatter` which is often required by TAPAS implementations within `transformers`. Use pip in a virtual environment:
    ```bash
    pip install transformers torch pandas torch-scatter
    ```
    * `transformers`: The core Hugging Face library.
    * `torch`: The deep learning framework backend (PyTorch).
    * `pandas`: Required for creating and handling the table data structure (DataFrame).
    * `torch-scatter`: Often a required dependency for TAPAS model computations. Install it explicitly just in case.

## Installation

1.  **Clone or Download:** Get the `run_table_qa.py` script onto your local machine.
2.  **Create Virtual Environment (Recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    *(Use `.\.venv\Scripts\activate` on Windows)*
3.  **Install Python Libraries:** Run the pip command from the Prerequisites section within your activated virtual environment.

## Usage

1.  **Configure Inputs (Table & Question):**
    * Open the `run_table_qa.py` script in a text editor.
    * **Table Data:**
        * Locate the section marked `# 1. Define the table data...`.
        * Modify the `data` dictionary and the resulting `pd.DataFrame(data)` to represent the table you want to query. Ensure column names are strings.
    * **Question:**
        * Locate the line: `question = "What time is the concert at RAC Arena?"`
        * Change the question text to be relevant to the table data you provided.

2.  **Run the Script:**
    * Open your terminal or command prompt.
    * Make sure your virtual environment is activated.
    * Navigate to the directory containing the script.
    * Execute the script using Python:
        ```bash
        python run_table_qa.py
        ```

## Expected Output

The script will print the table data and the question being asked. The final output will be the model's answer derived from analyzing the table: