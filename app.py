import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_data():
    embeddings = np.load("embeddings.npy")
    with open("documents.txt", "r", encoding="utf-8") as f:
        documents = [line.strip() for line in f if line.strip()]
    return embeddings, documents

model = load_model()
embeddings, documents = load_data()

def retrieve_top_k(query, embeddings, k=10):
    """Retrieve top-k most similar documents using cosine similarity."""
    query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    similarities = cosine_similarity(query_vec, embeddings)[0]

    top_k_indices = similarities.argsort()[-k:][::-1]
    return [(documents[i], similarities[i]) for i in top_k_indices]

# Streamlit UI
st.title("Information Retrieval using Document Embeddings")
# Input query
query = st.text_input("Enter your query:")
if st.button("Search"):
    results = retrieve_top_k(query, embeddings)
    # Display results
    st.write("### Top 10 Relevant Documents:")
    for doc, score in results:
        st.write(f"- **{doc.strip()}** (Score: {score:.4f})")
