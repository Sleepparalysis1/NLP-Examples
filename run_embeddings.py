# Import the SentenceTransformer class, utility functions, and torch
from sentence_transformers import SentenceTransformer, util
import torch

print("-------------------------------------------")
print("Hugging Face Ecosystem Example")
print("Task: Semantic Similarity Search")
print("Library: sentence-transformers")
print("Model: all-MiniLM-L6-v2")
print("-------------------------------------------")

# 1. Determine device (GPU or CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 2. Load the Sentence Transformer model
print("Loading Sentence Transformer model (may download on first run)...")
try:
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    print("Model loaded successfully.")

except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'sentence-transformers' and 'torch' (or 'tensorflow') are installed.")
    exit()

# 3. Define the 'corpus' of sentences to search within
#    Using context relevant to Friday afternoon in Perth
corpus_sentences = [
    "It's a sunny Friday afternoon here in Perth.",
    "Many people are planning their weekend activities now.",
    "Traffic leaving the city centre can be busy around 2 PM.",
    "A walk along the Swan River seems like a good idea later.",
    "The weather in Western Australia is lovely today.",
    "Commuters might face delays on the freeway system.",
    "Finding parking downtown might be difficult this afternoon."
]

# 4. Define the query sentence we want to find similar sentences for
#    Note it uses different words than the target sentences in the corpus.
query_sentence = "How is the road congestion in the city right now?"

print("\nCorpus Sentences:")
for i, s in enumerate(corpus_sentences):
    print(f"- \"{s}\"")
print(f"\nQuery Sentence: \"{query_sentence}\"")

# 5. Encode the corpus and the query to get embeddings
print("\nGenerating embeddings for corpus and query...")
try:
    # It's often efficient to encode the corpus once and reuse it
    # convert_to_tensor=True is recommended for util.semantic_search
    corpus_embeddings = model.encode(corpus_sentences, convert_to_tensor=True, device=device)
    query_embedding = model.encode(query_sentence, convert_to_tensor=True, device=device)
    print("Embeddings generated.")

    # 6. Perform semantic search using cosine similarity
    #    util.semantic_search finds the top_k most similar sentences
    print("\nPerforming semantic search...")
    # Find the top 3 most similar sentences in the corpus
    top_k = 3
    # returns a list of lists for each query, containing dicts with 'corpus_id' and 'score'
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  # Get the results for the first (and only) query

    print("Search complete.")

    # 7. Print the results
    print(f"\n--- Top {top_k} Most Similar Sentences to the Query ---")
    if not hits:
        print("No similar sentences found.")
    else:
        print(f"Query: \"{query_sentence}\"\n")
        for rank, hit in enumerate(hits):
            corpus_id = hit['corpus_id'] # Index of the sentence in corpus_sentences
            score = hit['score']      # Cosine similarity score

            print(f"Rank {rank+1}: Score: {score:.4f}")
            print(f"   Sentence: \"{corpus_sentences[corpus_id]}\"")
            print("-" * 15) # Separator

    print("-------------------------------------------------")

except Exception as e:
    print(f"Error during embedding generation or search: {e}")


print("\nExample finished.")