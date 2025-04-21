# Local Dialogue Simulation using Text Generation Pipeline

This script simulates a multi-turn dialogue locally using the `microsoft/DialoGPT-medium` model.

**Important Note:** As the dedicated `"conversational"` pipeline task in `transformers` appears to be deprecated or removed in recent versions, this example uses the standard `"text-generation"` pipeline. Conversation history is managed manually by concatenating previous turns into the prompt for the model, separated by the model's specific end-of-sentence token.

## Features

* Simulates multi-turn conversations locally using a text generation approach.
* Uses the `microsoft/DialoGPT-medium` model, fine-tuned for dialogue.
* Manages conversation history explicitly in the script's logic via prompt concatenation.
* Leverages the Hugging Face `transformers` library and its `"text-generation"` pipeline.
* Optionally utilizes GPU for faster response generation.

## Model Used

* **Dialogue Model:** `microsoft/DialoGPT-medium` (used via text-generation pipeline)

## Prerequisites

Before running the script, ensure you have the following installed:

1.  **Python:** Python 3.8 or later recommended. 
2.  **System Dependencies:** None specific beyond standard build tools.
3.  **Python Libraries:** Install using pip in a virtual environment.
    ```bash
    pip install transformers torch
    ```
    * `transformers`: The core Hugging Face library (including `pipeline` and `AutoTokenizer`).
    * `torch`: The deep learning framework backend (PyTorch).

## Installation

1.  **Clone or Download:** Get the `run_dialogue_generation.py` script onto your local machine.
2.  **Create Virtual Environment (Recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    *(Use `.\.venv\Scripts\activate` on Windows)*
3.  **Install Python Libraries:** Run the pip command from the Prerequisites section within your activated virtual environment.

## Usage

1.  **Configure Conversation (Optional):**
    * Open the `run_dialogue_generation.py` script.
    * You can modify the `user_inputs` list near the top to change the user's side of the pre-defined conversation turns.
2.  **Run the Script:**
    * Open your terminal or command prompt.
    * Make sure your virtual environment is activated.
    * Navigate to the directory containing the script.
    * Execute the script using Python:
        ```bash
        python run_dialogue_generation.py
        ```

## Expected Output

The script will print status messages and then display the simulated conversation turn by turn. Because history is manually concatenated into the prompt, the model *should* respond based on previous turns.


--- Starting Dialogue Simulation ---

User >>> Hi, how are you? Bot >>> I'm good, how are you?
User >>> What's a good plan for a Friday evening in Perth? Bot >>> I'm not sure, I'm not familiar with Perth.
User >>> That's okay. What do people usually do on Friday evenings? Bot >>> A lot of people go out to eat or drink with friends.
--- Dialogue Finished ---

*(**Note:** Responses depend heavily on the model's training data. DialoGPT might still give generic answers or get repetitive. The quality also depends on how well the history is formatted in the prompt, including the use of the EOS token).*

## Troubleshooting

* **Library Import Errors:** Ensure `transformers` and `torch` are installed correctly in the active environment.
* **Model Download Issues:** Check internet connection. DialoGPT-medium is several hundred MB.
* **Response Quality/Repetition:** This can happen with dialogue models. Try different phrasing for user inputs. Ensure the history concatenation and `eos_token` usage are correct. Experiment with generation parameters like `max_new_tokens`, `temperature`, `top_k`.

## Hardware Considerations

* **CPU:** Possible, but response generation will likely be noticeably slow for interactive use.
* **GPU:** Recommended for better performance.
* **RAM:** DialoGPT-medium requires a moderate amount of RAM.

## License

* The `run_dialogue_generation.py` script is provided as an example (consider MIT License).
* Hugging Face libraries (`transformers`) are typically Apache 2.0 licensed.
* The `microsoft/DialoGPT-medium` model is available under the MIT license.