from __future__ import annotations
 
import os
import re
from collections import Counter
 
import nltk
import requests
import torch
import chromadb

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from nltk.corpus import stopwords
from nltk.metrics import ConfusionMatrix
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import pipeline


 
load_dotenv()
 
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN", "").strip()
X_BASE = "https://api.twitter.com/2"
 
app = Flask(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sentiment_model = pipeline(
    task="sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=0 if DEVICE == "cuda" else -1
)

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=DEVICE
)

chroma_client = chromadb.Client()

qa_generator = pipeline(
    task="text2text-generation",
    model="google/flan-t5-small",
    device=0 if DEVICE == "cuda" else -1
)

print("Running models on:", DEVICE)
 
def ensure_nltk():
    try:
        _ = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")
    try:
        _ = word_tokenize("hello world")
    except LookupError:
        nltk.download("punkt")
        nltk.download("punkt_tab")
 
ensure_nltk()
EN_STOP = set(stopwords.words("english"))

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
WS_RE = re.compile(r"\s+")

def clean_text(text):
    text = text or ""
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    text = HASHTAG_RE.sub(r"\1", text)
    return WS_RE.sub(" ", text).strip()


def tokenize(text):
    words = word_tokenize(clean_text(text).lower())

    return [
        word
        for word in words
        if word.isalpha()
        and word not in EN_STOP
        and len(word) > 2
    ]


def preprocess_sentiment(text):
    words = []

    for word in (text or "").split():
        if word.startswith("@"):
            word = "@user"
        elif word.startswith("http"):
            word = "http"

        words.append(word)

    return " ".join(words)

def predict_sentiments(texts):
    if not texts:
        return []

    prepared = [
        preprocess_sentiment(text)
        for text in texts
    ]

    results = sentiment_model(
        prepared,
        truncation=True,
        max_length=512,
        batch_size=8
    )

    return [
        {
            "label": result["label"].lower(),
            "score": float(result["score"])
        }
        for result in results
    ]
 
 
def search_recent_tweets(query, max_results=10, lang="en"):
    if not X_BEARER_TOKEN:
        raise RuntimeError("Missing X_BEARER_TOKEN in .env")

    query = query.strip()

    if not query:
        return []

    search_query = f"({query}) -is:retweet"

    if lang:
        search_query += f" lang:{lang}"

    response = requests.get(
        f"{X_BASE}/tweets/search/recent",
        headers={
            "Authorization": f"Bearer {X_BEARER_TOKEN}"
        },
        params={
            "query": search_query,
            "max_results": max(10, min(int(max_results), 20)),
            "tweet.fields": "created_at,lang,public_metrics"
        },
        timeout=25
    )

    if not response.ok:
        raise RuntimeError(
            f"X API error {response.status_code}: "
            f"{response.text}"
        )

    return response.json().get("data", [])
 
def summarize(items):
    counts = Counter(
        item["label"]
        for item in items
    )

    scores = [
        float(item["score"])
        for item in items
    ]

    return {
        "n": len(items),
        "positive": counts["positive"],
        "negative": counts["negative"],
        "neutral": counts["neutral"],
        "avg_score": (
            sum(scores) / len(scores)
            if scores else 0.0
        )
    }
 
@app.get("/")
def home():
    return render_template("index.html")
 
@app.post("/api/analyze")
def api_analyze():
    body = request.get_json(silent=True) or {}

    query = (body.get("query") or "").strip()
    lang = (body.get("lang") or "en").strip()
    n = max(10, min(int(body.get("n", 10)), 20))

    if not query:
        return jsonify({
            "error": "query is required"
        }), 400

    try:
        tweets = search_recent_tweets(
            query,
            max_results=n,
            lang=lang
        )

        texts = [
            tweet.get("text", "")
            for tweet in tweets
        ]

        sentiments = predict_sentiments(texts)

        items = []
        corpus_tokens = []
        token_counter = Counter()

        for tweet, sentiment in zip(
            tweets,
            sentiments
        ):
            text = tweet.get("text", "")
            tokens = tokenize(text)

            corpus_tokens.append(tokens)
            token_counter.update(tokens)

            items.append({
                "id": tweet.get("id"),
                "created_at": tweet.get("created_at"),
                "lang": tweet.get("lang"),
                "text": text,
                "cleaned": clean_text(text),
                "label": sentiment["label"],
                "score": sentiment["score"]
            })

        query_tokens = tokenize(query)

        if corpus_tokens and query_tokens:
            bm25 = BM25Okapi(corpus_tokens)
            scores = bm25.get_scores(query_tokens)

            for item, score in zip(items, scores):
                item["bm25_score"] = float(score)
        else:
            for item in items:
                item["bm25_score"] = 0.0

        items.sort(
            key=lambda item: item["bm25_score"],
            reverse=True
        )

        top_tokens = [
            {
                "token": token,
                "count": count
            }
            for token, count
            in token_counter.most_common(10)
        ]

        return jsonify({
            "query": query,
            "summary": summarize(items),
            "top_tokens": top_tokens,
            "items": items
        })

    except Exception as exc:
        return jsonify({
            "error": str(exc)
        }), 500
    
def precision_recall_from_matrix(labels, matrix):
    k = len(labels)
    col_sums = [sum(matrix[r][c] for r in range(k)) for c in range(k)]
    row_sums = [sum(matrix[r][c] for c in range(k)) for r in range(k)]

    per_class = {}
    precisions = []
    recalls = []

    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fp = col_sums[i] - tp
        fn = row_sums[i] - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        per_class[label] = {"precision": precision, "recall": recall}
        precisions.append(precision)
        recalls.append(recall)
    macro_precision = sum(precisions) / k if k > 0 else 0.0
    macro_recall = sum(recalls) / k if k > 0 else 0

    return {"per_class": per_class, "macro_precision": macro_precision, "macro_recall": macro_recall}
 
@app.post("/api/evaluate")
def api_evaluate():
    body = request.get_json()
    items = body.get("items")
 
    labels = ["positive", "neutral", "negative"]
 
    y_true = []
    y_pred = []
    for item in items:
        t = (item.get("true_label") or "").strip() 
        p = (item.get("pred_label") or "").strip()
        if t in labels and p in labels:
            y_true.append(t)
            y_pred.append(p)
 
    if not y_true:
        return jsonify({"error": "No valid labeled items"}), 400
 
    cm = ConfusionMatrix(y_true, y_pred)
 
    matrix = []
    for r in labels:
        row = []
        for c in labels:
            row.append(int(cm[r, c]))
        matrix.append(row)
 
    total = sum(sum(r) for r in matrix)
    correct = sum(matrix[i][i] for i in range(len(labels)))
    acc = correct / total if total else 0.0
    pr = precision_recall_from_matrix(labels, matrix)
 
    return jsonify({
        "labels": labels,
        "matrix": matrix,
        "accuracy": acc,
        "precision_recall": pr
    })

@app.post("/api/qa")
def api_qa():
    body = request.get_json(silent=True) or {}

    question = (body.get("question") or "").strip()
    tweets = body.get("tweets") or []

    if not question:
        return jsonify({
            "error": "Question is required"
        }), 400

    documents = [
        (tweet.get("text") or "").strip()
        for tweet in tweets
        if (tweet.get("text") or "").strip()
    ]

    if not documents:
        return jsonify({
            "error": "No tweets available. Run Analyze first."
        }), 400

    try:
        collection_name = "tweet_qa"

        try:
            chroma_client.delete_collection(collection_name)
        except Exception:
            pass

        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        tweet_embeddings = embedding_model.encode(
            documents,
            normalize_embeddings=True
        ).tolist()

        collection.add(
            ids=[str(i) for i in range(len(documents))],
            documents=documents,
            embeddings=tweet_embeddings
        )

        question_embedding = embedding_model.encode(
            [question],
            normalize_embeddings=True
        ).tolist()

        result = collection.query(
            query_embeddings=question_embedding,
            n_results=min(3, len(documents)),
            include=["documents", "distances"]
        )

        retrieved = result["documents"][0]
        distances = result["distances"][0]

        context = "\n\n".join(
            f"Tweet {i + 1}: {text}"
            for i, text in enumerate(retrieved)
        )

        prompt = f"""
Answer the question using only the retrieved tweets.

If the tweets do not provide enough information, say:
The retrieved tweets do not provide enough information.

Retrieved tweets:
{context}

Question:
{question}

Answer:
""".strip()

        output = qa_generator(
            prompt,
            max_new_tokens=100,
            do_sample=False
        )

        sources = [
            {
                "text": text,
                "similarity": max(
                    0.0,
                    1.0 - float(distance)
                )
            }
            for text, distance in zip(retrieved, distances)
        ]

        return jsonify({
            "answer": output[0]["generated_text"].strip(),
            "sources": sources
        })

    except Exception as exc:
        return jsonify({
            "error": str(exc)
        }), 500
 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)