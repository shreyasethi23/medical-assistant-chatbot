# Medical Assistant Chatbot

A medical assistant chatbot that combines **chest X-ray image classification** with **medical question-answering** capabilities. This project integrates a Convolutional Neural Network (CNN) for pneumonia detection from X-ray images and a semantic text retrieval system based on Sentence Transformers and FAISS for answering medical queries.

---

## Project Structure
<img width="605" alt="image" src="https://github.com/user-attachments/assets/9e8964f9-0686-4261-a779-b9ef6e363451" />


medical-assistant-chatbot/
│
├── data/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── medical_qa.csv
│
├── models/
│   └── pneumonia_model.pth
│
├── notebooks/
│   ├── cnn_training.ipynb
│   └── test_predictions.csv
│
├── text_retrieval/
│   ├── build_index.py
│   ├── query.py
│   ├── qa_data.pkl
│   ├── qa_index.faiss
│   └── app.py

---

## How to Run

### 1. Train CNN Model

Train the pneumonia classifier using the Jupyter notebook:
jupyter notebook notebooks/cnn_training.ipynb

### 2. Build Text Retrieval Index

Build the FAISS index from the medical Q&A dataset: python3 text_retrieval/build_index.py --qa_csv data/medical_qa.csv --output_index text_retrieval/qa_index.faiss --output_data text_retrieval/qa_data.pkl

### 3. Run Query Script
Test text retrieval interactively: python3 text_retrieval/query.py --index_path text_retrieval/qa_index.faiss --data_path text_retrieval/qa_data.pkl

### 4. Run the Streamlit App
Launch the full Medical Assistant Chatbot interface: streamlit run app.py

