# Import the pipeline function from the transformers library
from transformers import pipeline
import torch # Import torch to ensure it's detected (if you installed it)

print("-------------------------------------------")
print("Hugging Face Local Inference Example")
print("Task: Sentiment Analysis")
print("-------------------------------------------")

# 1. Load the sentiment analysis pipeline
#    - The first time you run this, it will download the model files
#      (e.g., 'distilbert-base-uncased-finetuned-sst-2-english') and cache them locally.
#    - Subsequent runs will load the model directly from your local cache.
print("Loading model (may download on first run)...")
try:
    # device=0 forces GPU if available and torch is installed with CUDA support
    # device=-1 forces CPU
    # If you installed tensorflow, you might need different device handling or remove this arg.
    classifier = pipeline(
        "sentiment-analysis",
        # You can optionally specify the model:
        # model="distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if torch.cuda.is_available() else -1 # Use GPU if available, else CPU
    )
    print("Model loaded successfully.")
    if torch.cuda.is_available():
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU.")

except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'transformers' and 'torch' (or 'tensorflow') are installed.")
    exit()

# 2. Prepare your input data (a list of sentences)
sentences = [
    "Running AI models locally is quite empowering!",
    "I sometimes worry about the computational resources required.",
    "The weather in Perth today looks lovely.",
    "This example is easy to understand."
]
print(f"\nAnalyzing {len(sentences)} sentences...")

# 3. Run the inference
try:
    results = classifier(sentences)
    print("Analysis complete.")

    # 4. Print the results
    print("\n--- Results ---")
    for sentence, result in zip(sentences, results):
        label = result['label']
        score = result['score']
        print(f"Sentence: \"{sentence}\"")
        print(f"   -> Label: {label}, Score: {score:.4f}") # Format score to 4 decimal places
    print("---------------")

except Exception as e:
    print(f"Error during analysis: {e}")

print("\nExample finished.")