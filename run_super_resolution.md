# Local Image Super-Resolution (x2) with Swin2SR

This script performs image super-resolution (2x upscaling) locally using the `caidas/swin2SR-classical-sr-x2-64` model via the Hugging Face `transformers` library's `image-to-image` pipeline.

It takes an image file as input and generates an output image with twice the width and height, aiming for enhanced detail and sharpness compared to simple resizing. The upscaled image is saved as a PNG file.

It includes flexibility for the image input:
1.  It prioritizes using a local image file path specified within the script.
2.  If the specified file isn't found, it downloads a sample image (a city street scene) for demonstration.

## Features

* Performs image super-resolution (2x upscale) locally.
* Uses the `caidas/swin2SR-classical-sr-x2-64` model (Swin Transformer v2 based).
* Generates a higher-resolution version of the input image.
* Handles user-specified local image files with a fallback to a sample image.
* Saves the resulting upscaled image as a PNG file (`super_resolution_output.png`).
* Leverages the Hugging Face `transformers` library.
* Optionally utilizes GPU for faster processing.

## Model Used

* **Super-Resolution Model:** `caidas/swin2SR-classical-sr-x2-64`

## Prerequisites

Before running the script, ensure you have the following installed:

1.  **Python:** Python 3.8 or later recommended.
2.  **System Dependencies:** None specific beyond standard build tools.
3.  **Python Libraries:** Install using pip in a virtual environment. Standard vision libraries are needed.
    ```bash
    pip install transformers torch Pillow torchvision timm requests
    ```
    * `transformers`: The core Hugging Face library.
    * `torch`: The deep learning framework backend (PyTorch).
    * `Pillow`: For loading, handling, and saving images.
    * `torchvision`, `timm`: Often required/beneficial for vision models like Swin2SR.
    * `requests`: Used to download the sample image if needed.

## Installation

1.  **Clone or Download:** Get the `run_super_resolution.py` script onto your local machine.
2.  **Create Virtual Environment (Recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    *(Use `.\.venv\Scripts\activate` on Windows)*
3.  **Install Python Libraries:** Run the pip command from the Prerequisites section within your activated virtual environment.

## Usage

1.  **Configure Image Input:**
    * Open the `run_super_resolution.py` script in a text editor.
    * Locate the line: `user_image_path = "my_low_res_image.jpg"`
    * **Option A (Recommended):** Change the path to the *exact path* of the image file you want to upscale. For best results, use an image that isn't already extremely high resolution (e.g., 640x480, 1024x768).
    * **Option B:** Place your image file in the *same directory* as the script and name it `my_low_res_image.jpg`.
    * **Fallback:** If no file is found at `user_image_path`, the script downloads and uses the sample image (`sr_sample_image.jpg`).

2.  **Run the Script:**
    * Open your terminal or command prompt.
    * Make sure your virtual environment is activated.
    * Navigate to the directory containing the script.
    * Execute the script using Python:
        ```bash
        python run_super_resolution.py
        ```

## Expected Output

The script will print status messages, including the dimensions of the input image. The primary output is a saved image file:
* An **upscaled image** will be saved as **`super_resolution_output.png`** in the same directory.
* The script will print the dimensions of this output image, which should be 2x the width and 2x the height of the input image.
* Visually compare the `super_resolution_output.png` to the input image; it should appear larger and potentially sharper or more detailed (results vary depending on the input).

## Troubleshooting

* **File Not Found errors:** Double-check the `user_image_path`. Check internet connection if relying on the fallback. Ensure the image file is readable.
* **Library Import Errors:** Ensure all required libraries (`transformers`, `torch`, `Pillow`, `torchvision`, `timm`, `requests`) are installed.
* **Errors during Upscaling:** Ensure the input image file is valid (not corrupted). Very large input images might exceed available RAM or VRAM. Check console for specific errors (e.g., memory errors).
* **Output Quality:** Super-resolution quality depends heavily on the input image content, the model's capabilities, and the upscaling factor (fixed at 2x here). Artifacts can sometimes occur, especially on heavily compressed or noisy input images.

## Hardware Considerations

* **CPU:** Possible but likely **very slow** for image-to-image models like Swin2SR.
* **GPU:** NVIDIA GPU is **strongly recommended** for this task due to the computational cost.
* **RAM/VRAM:** Processing images, especially for upscaling, can be memory-intensive. Ensure sufficient system RAM and particularly GPU VRAM.

## License

* The `run_super_resolution.py` script is provided as an example (consider MIT License).
* Hugging Face libraries are typically Apache 2.0 licensed.
* The `caidas/swin2SR-classical-sr-x2-64` model license should be checked on its model card (often Apache 2.0 or similar permissive licenses).