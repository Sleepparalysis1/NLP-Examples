# Import the pipeline function and torch
from transformers import pipeline
import torch

print("-------------------------------------------")
print("Hugging Face Local Inference Example")
print("Task: Text Summarization")
print("Model: facebook/bart-large-cnn")
print("-------------------------------------------")

# 1. Load the summarization pipeline, explicitly specifying the BART model
print("Loading summarization model (may download on first run)...")
try:
    summarizer = pipeline(
        "summarization", # Specify the task
        model="facebook/bart-large-cnn", # Explicit model name
        device=0 if torch.cuda.is_available() else -1 # Use GPU if available, else CPU
        )
    print("Model loaded successfully.")
    if torch.cuda.is_available():
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU.")

except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'transformers', 'sentencepiece', and 'torch' (or 'tensorflow') are installed.")
    exit()

# 2. Define the longer text you want to summarize
#    Using context relevant to Friday afternoon in Perth
text_to_summarize = """
(Perth, WA - April 18, 2025) Commuters experienced significant delays across Perth's major arterial roads this Friday afternoon as multiple factors combined to slow traffic.
An earlier fender bender on the Kwinana Freeway northbound near the Narrows Bridge caused considerable backlog, although the site has now been cleared by emergency services.
Main Roads WA also reported heavier than usual volume on the Mitchell Freeway southbound approaching the city centre.
Adding to the congestion, several downtown streets including St Georges Terrace are partially closed for preparations related to the upcoming 'Lumiere Perth' light festival scheduled to begin next weekend.
Public transport users on Transperth services also faced minor delays on some bus routes navigating the affected areas.
Authorities are advising motorists heading out of the city for the weekend or just beginning their afternoon commute to allow for extra travel time, check live traffic updates via the Main Roads website, and consider alternative routes or delaying non-essential travel until after the peak period, expected around 4:00 PM to 5:30 PM AWST.
Event organisers reminded attendees for tonight's concert at RAC Arena to factor potential traffic delays into their travel plans.
"""

print(f"\nOriginal Text Length: {len(text_to_summarize)} characters")
# print(f"Original Text:\n\"{text_to_summarize}\"") # Uncomment to see the full original text


# 3. Define summarization parameters
#    These control the length of the generated summary.
min_summary_len = 30  # Minimum number of tokens in the summary
max_summary_len = 130 # Maximum number of tokens in the summary

print(f"\nGenerating summary (min length: {min_summary_len}, max length: {max_summary_len})...")

# 4. Run the summarization pipeline
try:
    # Setting do_sample=False encourages deterministic output (less random)
    summary_result = summarizer(
        text_to_summarize,
        max_length=max_summary_len,
        min_length=min_summary_len,
        do_sample=False
        )
    print("Summarization complete.")

    # 5. Print the result
    #    The result is a list containing a dictionary with the summary text
    print("\n--- Generated Summary ---")
    if summary_result:
        print(summary_result[0]['summary_text'])
    else:
        print("Could not generate summary.")
    print("-------------------------")

except Exception as e:
    print(f"Error during summarization: {e}")


print("\nExample finished.")