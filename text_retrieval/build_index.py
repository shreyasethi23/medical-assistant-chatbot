import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

# Load your dataset
csv_path = os.path.join(os.path.dirname(__file__), "../data/medical_qa.csv")
df = pd.read_csv(csv_path)

# Load pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode all questions
question_embeddings = model.encode(df['question'].tolist(), convert_to_numpy=True)

# Create and save FAISS index
dimension = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(question_embeddings)

# Save index and the DataFrame
faiss.write_index(index, "qa_index.faiss")
with open("qa_data.pkl", "wb") as f:
    pickle.dump(df, f)

print("Index built and saved successfully.")
