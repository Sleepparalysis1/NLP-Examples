# Local Image Segmentation with SegFormer

This script performs semantic image segmentation locally using the `nvidia/segformer-b0-finetuned-ade-512-512` model via the Hugging Face `transformers` library.

Given an input image, it predicts a class label for each pixel (e.g., road, sky, building, person, car) based on the categories in the ADE20K dataset. The script then saves a visualization overlaying colored masks for detected segments onto the original image.

It includes flexibility for the image input:
1.  It prioritizes using a local image file path specified within the script.
2.  If the specified file isn't found, it downloads a sample image (a city street scene) for demonstration.

## Features

* Performs semantic image segmentation locally (pixel-level classification).
* Uses the `nvidia/segformer-b0-finetuned-ade-512-512` model (SegFormer architecture).
* Assigns category labels to each pixel in the image (based on ADE20K dataset).
* Handles user-specified local image files with a fallback to a sample image.
* Saves a visualized output image (`segmentation_visualization.png`) showing colored segment masks and a legend.
* Leverages the Hugging Face `transformers` library and `matplotlib` for visualization.
* Optionally utilizes GPU for faster processing.

## Model Used

* **Image Segmentation Model:** `nvidia/segformer-b0-finetuned-ade-512-512` (fine-tuned on ADE20K)

## Prerequisites

Before running the script, ensure you have the following installed:

1.  **Python:** Python 3.8 or later recommended.
2.  **System Dependencies:** None specific beyond standard build tools.
3.  **Python Libraries:** Install using pip in a virtual environment. Includes libraries for model execution, image handling, web requests, array manipulation, and plotting/visualization.
    ```bash
    pip install transformers torch Pillow torchvision timm requests matplotlib numpy
    ```
    * `transformers`: The core Hugging Face library.
    * `torch`: The deep learning framework backend (PyTorch).
    * `Pillow`: For loading, handling, and saving images.
    * `torchvision`, `timm`: Often required/beneficial for vision models.
    * `requests`: Used to download the sample image if needed.
    * `matplotlib`: Used for creating and saving the visualization of the segmentation results.
    * `numpy`: Required for numerical operations, especially when handling image masks for visualization.

## Installation

1.  **Clone or Download:** Get the `run_segmentation.py` script onto your local machine.
2.  **Create Virtual Environment (Recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    *(Use `.\.venv\Scripts\activate` on Windows)*
3.  **Install Python Libraries:** Run the pip command from the Prerequisites section within your activated virtual environment.

## Usage

1.  **Configure Image Input:**
    * Open the `run_segmentation.py` script in a text editor.
    * Locate the line: `user_image_path = "my_segmentation_image.jpg"`
    * **Option A (Recommended):** Change the path to the *exact path* of the image file you want to segment (e.g., a street scene, landscape, indoor photo with distinct regions).
    * **Option B:** Place your image file in the *same directory* as the script and name it `my_segmentation_image.jpg`.
    * **Fallback:** If no file is found at `user_image_path`, the script downloads and uses the sample image (`segmentation_sample_image.jpg`).

2.  **Run the Script:**
    * Open your terminal or command prompt.
    * Make sure your virtual environment is activated.
    * Navigate to the directory containing the script.
    * Execute the script using Python:
        ```bash
        python run_segmentation.py
        ```

## Expected Output

The script will print status messages, including the image source used. The primary output is a saved image file:
* A visualization image named **`segmentation_visualization.png`** will be saved in the same directory.
* This image shows the original input image with **colored, semi-transparent masks** overlaid on the areas corresponding to the different semantic categories detected by the model (e.g., road, sky, car, person, building). A legend mapping the random colors to the detected labels is included next to the image.

You can open `segmentation_visualization.png` with any standard image viewer.
![my_segmentation_image](https://github.com/user-attachments/assets/b24ce646-2177-4d7d-952c-075c8b1fc23f)

![segmentation_visualization](https://github.com/user-attachments/assets/6a83ea7a-87b7-4fed-9f5a-a5fe855b93df)

## Troubleshooting

* **File Not Found errors:** Double-check the `user_image_path`. Check internet connection if relying on the fallback. Ensure the image file is readable.
* **Library Import Errors:** Ensure all required libraries (`transformers`, `torch`, `Pillow`, `torchvision`, `timm`, `requests`, `matplotlib`, `numpy`) are installed. Issues with `matplotlib` backends or font loading can sometimes occur depending on the system setup (the script includes a fallback for fonts).
* **Errors during Segmentation/Visualization:** Ensure the input image file is valid (not corrupted) and in a common format (JPG, PNG). Check console for errors. Very large images might consume significant RAM/VRAM during processing or visualization.
* **Visualization Quality:** The accuracy of the segmentation depends on the model and how well the image content matches the categories in the ADE20K dataset (which contains 150 categories).

## Hardware Considerations

* **CPU:** Possible but likely very slow for segmentation models like SegFormer.
* **GPU:** NVIDIA GPU strongly recommended for reasonable performance due to the pixel-level processing involved.
* **RAM/VRAM:** Segmentation models process images densely and can require significant memory, especially for high-resolution images.

## License

* The `run_segmentation.py` script is provided as an example (consider MIT License).
* Hugging Face libraries are typically Apache 2.0 licensed. `matplotlib` and `numpy` have their own permissive licenses.
* The `nvidia/segformer-b0-finetuned-ade-512-512` model is typically available under the Apache 2.0 license (check model card for specifics).
