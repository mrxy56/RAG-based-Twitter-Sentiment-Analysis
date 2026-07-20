const { useState } = React;
 
function badgeClass(label) {
  if (label === "positive") return "badge text-bg-success";
  if (label === "negative") return "badge text-bg-danger";
  return "badge text-bg-secondary";
}
 
function downloadCSV(filename, rows) {
  const headers = ["created_at", "bm25_score", "label", "score", "true_label", "text"];
  const escape = (v) => `"${String(v ?? "").replaceAll('"', '""').replaceAll("\n", " ")}"`;
 
  const lines = [
    headers.join(","),
    ...rows.map((r) => headers.map((h) => escape(r[h])).join(","))
  ];
 
  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
 
function App() {
  const [query, setQuery] = useState("Zohran Mamdani mayor Muslim NYC");
  const [n, setN] = useState(10);
  const [lang, setLang] = useState("en");
 
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [data, setData] = useState(null);
  const [evalRes, setEvalRes] = useState(null);

  const [question, setQuestion] = useState("");
  const [qaLoading, setQaLoading] = useState(false);
  const [qaResult, setQaResult] = useState(null);
 
  async function onAnalyze(e) {
    e.preventDefault();
    setLoading(true);
    setError("");
    setData(null);
    setEvalRes(null);
    setQaResult(null);
 
    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, n, lang })
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || "Analyze failed");
 
      json.items = (json.items || []).map((x) => ({ ...x, true_label: "" }));
      setData(json);
    } catch (err) {
      setError(err.message || "Analyze failed. Check backend logs.");
    } finally {
      setLoading(false);
    }
  }
 
  async function onEvaluate() {
    setEvalRes(null);
    try {
      const payload = (data.items || [])
        .filter((x) => x.true_label)
        .map((x) => ({ true_label: x.true_label, pred_label: x.label }));
 
      const res = await fetch("/api/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ items: payload })
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || "Evaluate failed");
      setEvalRes(json);
    } catch (err) {
      alert(err.message || "Evaluate failed");
    }
  }

  async function onAsk() {
  if (!question.trim()) {
    alert("Please enter a question.");
    return;
  }

  if (!data || !data.items || data.items.length === 0) {
    alert("Run Analyze first.");
    return;
  }

  setQaLoading(true);
  setQaResult(null);

  try {
    const res = await fetch("/api/qa", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        question,
        tweets: data.items
      })
    });

    const json = await res.json();

    if (!res.ok) {
      throw new Error(json.error || "QA failed");
    }

    setQaResult(json);

  } catch (err) {
    alert(err.message || "QA failed");
  } finally {
    setQaLoading(false);
  }
}
 
  return (
    <div className="container py-4">
      <div className="card shadow-sm">
        <div className="card-body">
          <h3 className="mb-3">Twitter Sentiment Analysis</h3>
 
          <form onSubmit={onAnalyze}>
            <div className="mb-3">
              <label className="form-label">Query</label>
              <input
                className="form-control"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                required
              />
            </div>
 
            <div className="row g-3 align-items-end">
              <div className="col-md-6">
                <label className="form-label">Number of Tweets (10~20)</label>
                <input
                  className="form-control"
                  type="number"
                  min="10"
                  max="20"
                  value={n}
                  onChange={(e) => setN(+e.target.value)}
                />
              </div>
 
              <div className="col-md-6">
                <button disabled={loading} className="btn btn-primary w-100" type="submit">
                  {loading ? "Analyzing..." : "Analyze"}
                </button>
              </div>
            </div>
          </form>
 
          {error && <div className="text-danger mt-3">{error}</div>}
 
          {data && (
            <>
              <hr className="my-4" />
 
              <div className="row g-3 align-items-end">
                <div className="col-md-8">
                  <h5>Summary</h5>
                  <div className="text-muted">
                    <b>Number of Tweets</b> {data.summary.n} ·{" "}
                    <b>Positive</b> {data.summary.positive} ·{" "}
                    <b>Negative</b> {data.summary.negative} ·{" "}
                    <b>Neutral</b> {data.summary.neutral} ·{" "}
                    <b>AVG</b> {Number(data.summary.avg_score).toFixed(3)}
                  </div>
                </div>
 
                <div className="col-md-4">
                  <button
                    className="btn btn-outline-secondary w-100"
                    type="button"
                    onClick={() => downloadCSV("tweets_sentiment.csv", data.items)}
                  >
                    Download CSV
                  </button>
                </div>
              </div>
 
              <div className="mt-4">
                <h5>Top tokens</h5>
                <div className="mt-2 d-flex flex-wrap gap-2">
                  {(data.top_tokens || []).map((t) => (
                    <span key={t.token} className="badge text-bg-light border">
                      {t.token} <span className="text-muted ms-1">{t.count}</span>
                    </span>
                  ))}
                </div>
              </div>

              <div className="mt-4">
                <h5>RAG Question Answering</h5>
        
                <div className="input-group">
                  <input
                    className="form-control"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="Ask a question about the retrieved tweets"
                  />

                <button
                  className="btn btn-primary"
                  type="button"
                  onClick={onAsk}
                  disabled={qaLoading}
                >
                  {qaLoading ? "Answering..." : "Ask"}
                </button>
              </div>

              {qaResult && (
                <div className="alert alert-light border mt-3">
                  <strong>Answer</strong>

                  <div className="mt-2">
                    {qaResult.answer}
                  </div>

                  <strong className="d-block mt-3">
                    Retrieved Tweets
                  </strong>

                  {(qaResult.sources || []).map((source, index) => (
                    <div
                      key={index}
                      className="border rounded p-2 mt-2 bg-white"
                    >
                      <small className="text-muted">
                          Similarity:{" "}
                          {Number(source.similarity).toFixed(3)}
                      </small>

                      <div>{source.text}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
 
              <div className="mt-4">
                <div className="d-flex flex-wrap align-items-center justify-content-between gap-2">
                  <div>
                    <h5 className="mb-0">Evaluation</h5>
                    <div className="text-muted">Select true labels below, then click Evaluate.</div>
                  </div>
 
                  <button className="btn btn-success" type="button" onClick={onEvaluate}>
                    Evaluate
                  </button>
                </div>
 
                {evalRes && (
                  <div className="mt-3">
                    <div className="text-muted mb-2">
                      <b>Accuracy</b> {Number(evalRes.accuracy).toFixed(3)} ·{" "}
                      <b>Precision</b> {Number(evalRes.precision_recall.macro_precision).toFixed(3)} ·{" "}
                      <b>Recall</b> {Number(evalRes.precision_recall.macro_recall).toFixed(3)}
                    </div>
 
                    <div className="table-responsive">
                      <table className="table table-bordered align-middle cm-table">
                        <thead className="table-light">
                          <tr>
                            <th>True \ Pred</th>
                            {evalRes.labels.map((l) => <th key={l}>{l}</th>)}
                          </tr>
                        </thead>
                        <tbody>
                          {evalRes.labels.map((r, i) => (
                            <tr key={r}>
                              <th className="table-light">{r}</th>
                              {evalRes.matrix[i].map((v, j) => (
                                <td key={j} className="text-center">{v}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
 
              <div className="mt-4">
                <h5>Tweets</h5>
 
                <div className="mt-3 d-flex flex-column gap-3">
                  {(data.items || []).map((it) => (
                    <div key={it.id} className="card border">
                      <div className="card-body">
                        <div className="d-flex flex-wrap gap-2 align-items-center mb-2">
                          <span className={badgeClass(it.label)}>{it.label}</span>
 
                          <span className="text-muted">
                            Sentiment Score: {Number(it.score).toFixed(3)}
                          </span>
 
                          <span className="text-muted">
                            BM25 Score: {Number(it.bm25_score ?? 0).toFixed(3)}
                          </span>
 
                          <span className="text-muted">
                            {(it.created_at || "").replace("T", " ").replace(".000Z", "Z")}
                          </span>
 
                          <div className="ms-auto" style={{ minWidth: "220px" }}>
                            <select
                              className="form-select"
                              value={it.true_label}
                              onChange={(e) => {
                                const v = e.target.value;
                                setData((prev) => ({
                                  ...prev,
                                  items: prev.items.map((x) =>
                                    x.id === it.id ? { ...x, true_label: v } : x
                                  )
                                }));
                              }}
                            >
                              <option value="">True Label</option>
                              <option value="positive">positive</option>
                              <option value="neutral">neutral</option>
                              <option value="negative">negative</option>
                            </select>
                          </div>
                        </div>
 
                        <div style={{ whiteSpace: "pre-wrap", lineHeight: "1.35" }}>
                          {it.text}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
 
ReactDOM.createRoot(document.getElementById("root")).render(<App />);