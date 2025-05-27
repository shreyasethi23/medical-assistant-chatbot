import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load FAISS index and data once
def load_resources():
    base_path = os.path.dirname(__file__)
    index_path = os.path.join(base_path, "qa_index.faiss")
    data_path = os.path.join(base_path, "qa_data.pkl")
    
    index = faiss.read_index(index_path)
    with open(data_path, "rb") as f:
        df = pickle.load(f)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return index, df, model

def query_index(query, index, df, model, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        question = df.iloc[idx]['question']
        answer = df.iloc[idx]['answer']
        results.append({'distance': dist, 'question': question, 'answer': answer})
    return results

if __name__ == "__main__":
    index, df, model = load_resources()
    user_query = input("Query: ")
    results = query_index(user_query, index, df, model, top_k=5)
    
    for i, res in enumerate(results, 1):
        print(f"Result {i}: (Distance: {res['distance']:.4f})")
        print(res['question'])
        print(res['answer'])
        print("--------------------------------------------------")