import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import os

# --- Setup ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your saved pneumonia classification model
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load("models/pneumonia_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = ['NORMAL', 'PNEUMONIA']

# Load FAISS index and question-answer data
index = faiss.read_index("text_retrieval/qa_index.faiss")
with open("text_retrieval/qa_data.pkl", "rb") as f:
    qa_df = pickle.load(f)

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Streamlit UI ---
st.title("Medical Assistant Chatbot")

st.header("1. Pneumonia Image Classification")
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
        pred_class = class_names[pred.item()]
        st.success(f"Prediction: {pred_class}")

st.header("2. Medical Q&A Text Retrieval")
query = st.text_input("Ask a medical question:")

if query:
    st.write(f"Query: {query}")
    query_embedding = sentence_model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_embedding, k=3)  # top 3 results

    for rank, (dist, idx) in enumerate(zip(D[0], I[0]), start=1):
        question = qa_df.iloc[idx]['question']
        answer = qa_df.iloc[idx]['answer']
        st.markdown(f"**Result {rank}** (Distance: {dist:.4f})")
        st.markdown(f"**Q:** {question}")
        st.markdown(f"**A:** {answer}")
        st.markdown("---")