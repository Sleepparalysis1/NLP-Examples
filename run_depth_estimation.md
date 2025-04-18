# Local Monocular Depth Estimation with DPT

This script performs monocular depth estimation locally using the `Intel/dpt-large` model via the Hugging Face `transformers` library. It takes a single RGB image as input and outputs a predicted depth map, saving it as an image file.

Monocular depth estimation predicts the distance of objects from the camera using only a single 2D image, which is an inherently challenging task. The output is a relative depth map.

It includes flexibility for the image input:
1.  It prioritizes using a local image file path specified within the script.
2.  If the specified file isn't found, it downloads a sample image (a city street scene) for demonstration.

## Features

* Performs monocular (single image) depth estimation locally.
* Uses the `Intel/dpt-large` model (Dense Prediction Transformer).
* Generates a depth map representing estimated distances.
* Handles user-specified local image files with a fallback to a sample image.
* Saves the resulting depth map as a PNG image file (`depth_map_output.png`).
* Leverages the Hugging Face `transformers` library.
* Optionally utilizes GPU for faster processing.

## Model Used

* **Depth Estimation Model:** `Intel/dpt-large`

## Prerequisites

Before running the script, ensure you have the following installed:

1.  **Python:** Python 3.8 or later recommended.
2.  **System Dependencies:** None specific beyond standard build tools.
3.  **Python Libraries:** Install using pip in a virtual environment. DPT models often rely on `timm` and require standard vision libraries.
    ```bash
    pip install transformers torch Pillow torchvision timm requests
    ```
    * `transformers`: The core Hugging Face library.
    * `torch`: The deep learning framework backend (PyTorch).
    * `Pillow`: For loading, handling, and saving images.
    * `torchvision`, `timm`: Often required/beneficial for vision models (like DPT backbones) and image preprocessing.
    * `requests`: Used to download the sample image if needed.

## Installation

1.  **Clone or Download:** Get the `run_depth_estimation.py` script onto your local machine.
2.  **Create Virtual Environment (Recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    *(Use `.\.venv\Scripts\activate` on Windows)*
3.  **Install Python Libraries:** Run the pip command from the Prerequisites section within your activated virtual environment.

## Usage

1.  **Configure Image Input:**
    * Open the `run_depth_estimation.py` script in a text editor.
    * Locate the line: `user_image_path = "my_depth_image.jpg"`
    * **Option A (Recommended):** Change the path `"my_depth_image.jpg"` to the *exact path* of the image file you want to estimate depth for (e.g., a landscape, street scene, or indoor photo with varying distances). Avoid flat images with little depth variation.
    * **Option B:** Place your image file in the *same directory* as the script and name it `my_depth_image.jpg`.
    * **Fallback:** If no file is found at `user_image_path`, the script downloads and uses the sample image (`depth_sample_image.jpg`).

2.  **Run the Script:**
    * Open your terminal or command prompt.
    * Make sure your virtual environment is activated.
    * Navigate to the directory containing the script.
    * Execute the script using Python:
        ```bash
        python run_depth_estimation.py
        ```

## Expected Output

The script will print status messages, including the image source used. The primary output is not text, but an image file:
* A **predicted depth map image** will be saved as **`depth_map_output.png`** in the same directory.
* This output image will typically be grayscale. Pixel intensity represents estimated depth (the exact mapping - lighter=closer or darker=closer - can vary between models, but relative depth should be apparent).

You can open `depth_map_output.png` with any standard image viewer to see the result.

## Troubleshooting

* **File Not Found errors:** Double-check the `user_image_path`. Check internet connection if relying on the fallback. Ensure the image file is readable.
* **Library Import Errors:** Ensure all required libraries (`transformers`, `torch`, `Pillow`, `torchvision`, `timm`, `requests`) are installed.
* **Errors during Estimation:** Ensure the input image file is valid (not corrupted) and in a common format (JPG, PNG). Check console for specific errors.
* **Output Quality:** The quality of the depth map depends heavily on the input image, the scene's complexity, lighting, and the model's capabilities. Monocular depth estimation provides relative, not absolute, depth and can have artifacts.

## Hardware Considerations

* **CPU:** Possible but likely very slow for DPT-Large.
* **GPU:** NVIDIA GPU strongly recommended for this task due to model size and computation.
* **RAM:** DPT-Large is a sizable model; ensure sufficient RAM (16GB+ might be needed for comfortable operation).

## License

* The `run_depth_estimation.py` script is provided as an example (consider MIT License).
* Hugging Face libraries are typically Apache 2.0 licensed.
* The `Intel/dpt-large` model is typically available under the MIT license (check model card for specifics).