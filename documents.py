import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


documents = [
    "Python is widely used for data analysis and machine learning.",
    "Java is a popular language for enterprise applications.",
    "Cybersecurity protects systems from digital attacks.",
    "Cloud computing enables scalable online services.",
    "Databases store and manage structured information.",
    "Artificial intelligence powers smart applications.",
    "Web development involves frontend and backend programming.",
    "Blockchain technology ensures secure digital transactions.",
    "Mobile apps are developed for Android and iOS platforms.",
    "Software engineering focuses on building reliable systems.",
    "Machine learning algorithms improve predictive analytics.",
    "DevOps integrates development and operations workflows.",
    "Virtual reality offers immersive digital experiences.",
    "Augmented reality overlays digital content onto the real world.",
    "Natural language processing allows computers to understand text.",
    "Computer vision enables machines to see and interpret images.",
    "Internet of Things connects everyday devices to the internet.",
    "Big data analytics processes massive datasets efficiently.",
    "Quantum computing uses quantum bits for complex calculations.",
    "Edge computing processes data closer to the source.",
    "5G networks provide faster and more reliable internet connections.",
    "Robotic process automation automates repetitive tasks.",
    "Containerization helps deploy applications consistently.",
    "Microservices architecture improves software scalability.",
    "Open-source software encourages collaboration and innovation.",
    "Data mining extracts meaningful patterns from raw data.",
    "Encryption secures sensitive information from unauthorized access.",
    "Digital marketing uses data to target specific audiences.",
    "E-commerce platforms facilitate online buying and selling.",
    "Cloud storage allows users to save data remotely.",
    "High-performance computing accelerates scientific research.",
    "Autonomous vehicles rely on AI and sensors for navigation.",
    "Speech recognition converts spoken words into text.",
    "Facial recognition identifies individuals based on images.",
    "Wearable technology monitors health and fitness metrics.",
    "Smart home devices enable automated control of household systems.",
    "Programming languages evolve to support modern software needs.",
    "Software testing ensures applications work as intended.",
    "API integration connects different software systems.",
    "Augmented analytics enhances business intelligence with AI.",
    "Graph databases store complex relationships between entities.",
    "Cloud-native applications are designed for cloud environments.",
    "Serverless computing runs code without managing servers.",
    "Digital twins simulate physical systems for analysis.",
    "Deep learning models mimic neural networks for AI tasks.",
    "Reinforcement learning trains agents via rewards and penalties.",
    "Data visualization communicates insights effectively.",
    "Content management systems organize and publish digital content.",
    "Gamification increases user engagement in applications.",
    "Cyber-physical systems integrate computing with physical processes.",
    "Edge AI brings artificial intelligence to edge devices."
]

# Save documents to file
with open("documents.txt", "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc + "\n")


vectorizer = TfidfVectorizer(stop_words="english")
doc_vectors = vectorizer.fit_transform(documents)  # sparse matrix

np.save("embeddings.npy", doc_vectors.toarray())  # embeddings as NumPy array

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)  # save vectorizer for queries

print("TF-IDF embeddings saved as embeddings.npy")
print("Vectorizer saved as vectorizer.pkl")
