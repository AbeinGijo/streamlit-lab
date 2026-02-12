import numpy as np
from sentence_transformers import SentenceTransformer

# Load documents
with open("documents.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f if line.strip()]

model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
embeddings = model.encode(documents, convert_to_numpy=True, normalize_embeddings=True)

# Save embeddings
np.save("embeddings.npy", embeddings)

print("Saved embeddings.npy with shape:", embeddings.shape)
