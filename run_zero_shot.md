# Hugging Face Local Inference Example: Zero-Shot Classification

This script demonstrates how to perform zero-shot image classification locally using the `openai/clip-vit-base-patch32` model and the `transformers` library. 

**Zero-shot classification** allows us to classify an image *without* having explicitly trained the model on the specific labels we're interested in. Instead, we provide a list of potential labels at inference time, and the model determines how well the image matches each label based on its learned understanding of image-text relationships.

## Key Features

* **Local Inference:** Perform classification directly on your machine.
* **Zero-Shot Capability:** Classify images against any list of labels you provide.
* **CLIP Model:** Utilizes the powerful `openai/clip-vit-base-patch32` model for image-text understanding.
* **User-Friendly:** Easily customize input image and candidate labels.
* **GPU Support:** Optionally leverage a GPU for significantly faster processing.

## Prerequisites

* **Python:** Python 3.8 or later is recommended.
* **System Dependencies:** None specific beyond standard build tools.
* **Python Libraries:**
    ```bash
    pip install transformers torch Pillow torchvision timm requests ftfy regex
    ```
    * `transformers`: Core Hugging Face library for model loading and inference.
    * `torch`: PyTorch for model execution.
    * `Pillow`: For image loading and handling.
    * `torchvision`, `timm`: Often required or beneficial for vision models.
    * `requests`: For downloading a sample image if needed.
    * `ftfy`, `regex`: Text processing libraries often used by CLIP models.

## Installation

1. **Clone or Download:** Obtain the `run_zero_shot.py` script.
2. **Create Virtual Environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate 
    ```
3. **Install Dependencies:**
    ```bash
    pip install transformers torch Pillow torchvision timm requests ftfy regex
    ```

## Usage

1. **Configure Inputs:**
    * **Image:**
        * **Option 1:** **Use a local image:** * Modify `user_image_path` in the script to the path of your image.
        * **Option 2:** **Use a sample image:** If `user_image_path` is not found, the script will download a sample image.
    * **Candidate Labels:** * Modify the `candidate_labels` list with the labels you want to use for classification.

2. **Run the Script:**
    ```bash
    python run_zero_shot.py
    ```

## Expected Output

* The script will print a list of candidate labels ranked by their predicted relevance to the image, along with their scores. 

--- Zero-Shot Classification Results --- Rank 1: Score: 0.9921, Label: cats sleeping Rank 2: Score: 0.0035, Label: furniture Rank 3: Score: 0.0028, Label: remote control Rank 4: Score: 0.0008, Label: beach scene Rank 5: Score: 0.0005, Label: city buildings Rank 6: Score: 0.0003, Label: dogs playing fetch

## Notes

* The accuracy of zero-shot classification depends heavily on the quality of the image, the chosen model, and the relevance of the provided labels.
* GPU acceleration is highly recommended for faster processing.

## License

* This script is provided under the MIT License.
* The `openai/clip-vit-base-patch32` model is available under the MIT license.
* Hugging Face libraries (`transformers`, `datasets`, etc.) are typically licensed under Apache 2.0. 