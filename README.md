# Twitter Sentiment RAG

A simple Twitter sentiment analysis and question-answering project using IBM Watson NLU, sentence embeddings, ChromaDB, and RAG.

## Features

- Search recent tweets with the X API
- Analyze sentiment as positive, neutral, or negative
- Store tweet embeddings in ChromaDB
- Retrieve relevant tweets with semantic search
- Answer questions based on retrieved tweets
- Evaluate sentiment classification and retrieval quality

## Technologies

- Python
- Google Colab
- X API
- IBM Watson NLU
- Sentence Transformers
- ChromaDB
- Qwen
- NLTK

## Workflow

```text
X API
  ↓
Tweet preprocessing
  ↓
Sentiment analysis
  ↓
Sentence embeddings
  ↓
ChromaDB
  ↓
Semantic retrieval
  ↓
RAG question answering
```

## Installation

```python
!pip install -q requests nltk ibm-watson chromadb sentence-transformers transformers accelerate
```

## API Keys

```python
X_BEARER_TOKEN = "YOUR_X_BEARER_TOKEN"
IBM_NLU_APIKEY = "YOUR_IBM_API_KEY"
IBM_NLU_URL = "YOUR_IBM_SERVICE_URL"
```

Do not upload real API keys to GitHub.

## Example

```python
query = "Tesla"

raw_tweets = search_tweets(query, max_results=10)
tweets = analyze_tweets(raw_tweets)

store_tweets(
    tweets=tweets,
    topic=query
)

result = answer_question(
    question="What are the main complaints in these tweets?",
    top_k=5,
    topic=query
)

print(result["answer"])
```

## Evaluation

Sentiment evaluation includes:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Retrieval evaluation includes:

- Precision@K
- Recall@K
- MRR
