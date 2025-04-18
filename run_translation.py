# Import the pipeline function and torch
from transformers import pipeline
import torch

print("-------------------------------------------")
print("Hugging Face Local Inference Example")
print("Task: Translation (English to French)")
print("Model: Helsinki-NLP/opus-mt-en-fr")
print("-------------------------------------------")

# 1. Load the translation pipeline
#    - Specify the task as "translation_xx_to_yy" where xx/yy are language codes.
#    - Explicitly provide the Helsinki-NLP model name.
print("Loading translation model (may download on first run)...")
try:
    # Note the task format: translation_{source_lang}_to_{target_lang}
    translator = pipeline(
        "translation_en_to_fr", # Task for English to French
        model="Helsinki-NLP/opus-mt-en-fr", # Explicit model name
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

# 2. Define the English text you want to translate
#    Using context relevant to Friday afternoon in Perth
text_to_translate = "It is a beautiful Friday afternoon here in Perth, Western Australia. Perhaps I will go for a walk by the Swan River later today."

print(f"\nOriginal English Text:\n\"{text_to_translate}\"")


# 3. Run the translation pipeline
print("\nTranslating to French...")
try:
    # The pipeline handles tokenization, translation, and decoding
    translation_result = translator(text_to_translate)
    print("Translation complete.")

    # 4. Print the result
    #    The result is a list containing a dictionary. The translated text
    #    is typically under the 'translation_text' key.
    print("\n--- Generated French Translation ---")
    if translation_result:
        print(translation_result[0]['translation_text'])
    else:
        print("Could not generate translation.")
    print("----------------------------------")

except Exception as e:
    print(f"Error during translation: {e}")


print("\nExample finished.")