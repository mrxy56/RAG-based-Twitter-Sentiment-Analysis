# RAG-based Twitter Sentiment Analysis

A Flask and React application for Twitter sentiment analysis and RAG question answering.

## Features

- Retrieve recent tweets with the X API
- Rank tweets with BM25
- Classify sentiment as positive, neutral, or negative
- Store tweet embeddings in ChromaDB
- Retrieve relevant tweets with semantic search
- Answer questions using retrieved tweets
- Evaluate predictions with Accuracy, Precision, Recall, and a Confusion Matrix
- Export results as CSV

## Models

- Sentiment: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Generation: `google/flan-t5-small`

## Workflow

```text
X API
  ↓
Tweet preprocessing
  ↓
Twitter-RoBERTa sentiment analysis
  ↓
BM25 ranking
  ↓
React interface
  ├── Manual evaluation
  └── RAG QA
        ↓
     MiniLM embeddings
        ↓
     ChromaDB retrieval
        ↓
     FLAN-T5 answer
```

## Project Structure

```text
RAG-based-Twitter-Sentiment-Analysis/
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── .env
│   ├── .env.example
│   ├── templates/index.html
│   └── static/
│       ├── app.jsx
│       └── styles.css
├── .gitignore
└── README.md
```

## Installation

```bash
cd backend
python -m pip install -r requirements.txt
```

Create `backend/.env`:

```env
X_BEARER_TOKEN=YOUR_X_BEARER_TOKEN
HF_TOKAN=YOUR_HF_TOKEN
```

## Run

```bash
cd backend
python app.py
```

Open:

```text
http://127.0.0.1:5001
```

## Main API Endpoints

- `POST /api/analyze` — retrieve, classify, and rank tweets
- `POST /api/qa` — answer questions from retrieved tweets
- `POST /api/evaluate` — calculate sentiment evaluation metrics

## Model Size

Approximate model weights:

| Model | Size |
|---|---:|
| Twitter-RoBERTa | 500 MB |
| MiniLM | 90 MB |
| FLAN-T5-small | 300 MB |
