from __future__ import annotations
 
import os
import re
from collections import Counter
 
import nltk
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from nltk.corpus import stopwords
from nltk.metrics import ConfusionMatrix
from nltk.tokenize import word_tokenize
 
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import (
    Features,
    NaturalLanguageUnderstandingV1,
    SentimentOptions,
)
from rank_bm25 import BM25Okapi
 
load_dotenv()
 
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN", "").strip()
IBM_NLU_APIKEY = os.getenv("IBM_NLU_APIKEY", "").strip()
IBM_NLU_URL = os.getenv("IBM_NLU_URL", "").strip()
IBM_NLU_VERSION = os.getenv("IBM_NLU_VERSION", "2022-04-07").strip()
 
app = Flask(__name__)
 
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
 
def clean_text(s):
    s = s or ""
    s = URL_RE.sub(" ", s)
    s = MENTION_RE.sub(" ", s)
    s = HASHTAG_RE.sub(r"\1", s) 
    s = WS_RE.sub(" ", s).strip()
    return s
 
def tokens_from_text(s):
    s = clean_text(s).lower()
    toks = word_tokenize(s)
    return [t for t in toks if t.isalpha() and t not in EN_STOP and len(t) > 2]
 

def make_nlu():
    if not IBM_NLU_APIKEY or not IBM_NLU_URL:
        raise RuntimeError("Missing IBM_NLU_APIKEY / IBM_NLU_URL in backend/.env")
    authenticator = IAMAuthenticator(IBM_NLU_APIKEY)
    nlu = NaturalLanguageUnderstandingV1(version=IBM_NLU_VERSION, authenticator=authenticator)
    nlu.set_service_url(IBM_NLU_URL)
    return nlu
 
NLU = make_nlu()

def watson_sentiment(text):
    global NLU
 
    resp = NLU.analyze(
        text=text,
        features=Features(sentiment=SentimentOptions())
    ).get_result()
 
    doc = (resp.get("sentiment", {}) or {}).get("document", {}) or {}
    return {
        "label": doc.get("label", "neutral"),
        "score": float(doc.get("score", 0.0)),
    }
 

X_BASE = "https://api.twitter.com/2"
 
def x_search_recent(query, max_results=10, lang="en"):
    if not X_BEARER_TOKEN:
        raise RuntimeError("Missing X_BEARER_TOKEN in backend/.env")
 
    headers = {"Authorization": f"Bearer {X_BEARER_TOKEN}"}
    q = (query or "").strip()
    if not q:
        return []
 
    if lang:
        q = f"({q}) lang:{lang} -is:retweet"
    else:
        q = f"({q}) -is:retweet"
 
    params = {
        "query": q,
        "max_results": max(10, min(int(max_results), 20)),
        "tweet.fields": "created_at,lang,public_metrics",
    }
 
    r = requests.get(f"{X_BASE}/tweets/search/recent", headers=headers, params=params, timeout=25)
    if r.status_code >= 400:
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise RuntimeError(f"X API error {r.status_code}: {detail}")
 
    data = r.json()
    return data.get("data", []) or []
 
def summarize(items):
    n = len(items)
    if n == 0:
        return {"n": 0, "positive": 0, "negative": 0, "neutral": 0, "avg_score": 0.0}
    pos = sum(1 for x in items if x["label"] == "positive")
    neg = sum(1 for x in items if x["label"] == "negative")
    neu = n - pos - neg
    avg = sum(float(x["score"]) for x in items) / n
    return {"n": n, "positive": pos, "negative": neg, "neutral": neu, "avg_score": avg}
 
@app.get("/")
def home():
    return render_template("index.html")
 
@app.post("/api/analyze")
def api_analyze():
    body = request.get_json()
    query = (body.get("query") or "").strip()
    n = int(body.get("n"))      
    lang = (body.get("lang") or "en").strip()
 
    if not query:
        return jsonify({"error": "query is required"}), 400
 
    n = max(5, min(n, 20))
 
    try:
        tweets = x_search_recent(query, max_results=n, lang=lang)
        token_counter = Counter()
        items = []
        corpus_tokens = []
 
        for t in tweets:
            raw = t.get("text", "")
            cleaned = clean_text(raw)
            toks = tokens_from_text(raw)
            token_counter.update(toks)
            corpus_tokens.append(toks)
 
            s = watson_sentiment(cleaned)
            items.append({
                "id": t.get("id"),
                "created_at": t.get("created_at"),
                "lang": t.get("lang"),
                "text": raw,
                "cleaned": cleaned,
                "label": s["label"],
                "score": s["score"],
            })

        q_tokens = tokens_from_text(query)
        if corpus_tokens and q_tokens:
            bm25 = BM25Okapi(corpus_tokens)
            scores = bm25.get_scores(q_tokens)
            for item, score in zip(items, scores):
                item["bm25_score"] = float(score)
            items.sort(key=lambda x: x.get("bm25_score", 0.0), reverse=True)
        else:
            for item in items:
                item["bm25_score"] = 0.0
 
        top_tokens = [{"token": w, "count": c} for w, c in token_counter.most_common(10)]
 
        return jsonify({
            "query": query,
            "summary": summarize(items),
            "top_tokens": top_tokens,
            "items": items
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
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
 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)