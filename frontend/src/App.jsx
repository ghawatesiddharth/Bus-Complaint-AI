import React, { useState } from "react";
import {
  AlertTriangle,
  ArrowRight,
  Bus,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Clock3,
  MessageSquareText,
  RefreshCw,
  Search,
  ShieldAlert,
  Sparkles,
  ThumbsDown,
  ThumbsUp,
  WalletCards,
  XCircle,
} from "lucide-react";

const API_URL = "https://bus-complaint-ai-api.onrender.com";

const SAMPLE_COMPLAINTS = [
  "My bus was supposed to arrive at 8 PM but it arrived two hours late.",
  "I was charged twice for my bus ticket and I have not received my refund.",
  "The driver was driving dangerously and we almost had an accident. This was very unsafe.",
  "My bus was two hours late and customer support did not respond. I want a refund.",
];

function BusLogo() {
  return (
    <div className="brand-mark">
      <Bus size={25} strokeWidth={2.2} />
    </div>
  );
}

function StatIcon({ type }) {
  if (type === "department") {
    return <MessageSquareText size={20} />;
  }

  if (type === "sentiment") {
    return <ThumbsDown size={20} />;
  }

  if (type === "urgency") {
    return <AlertTriangle size={20} />;
  }

  return <Sparkles size={20} />;
}

function getSentimentIcon(sentiment) {
  if (!sentiment) return <MessageSquareText size={20} />;

  const value = sentiment.toLowerCase();

  if (value.includes("positive")) {
    return <ThumbsUp size={20} />;
  }

  if (value.includes("negative")) {
    return <ThumbsDown size={20} />;
  }

  return <MessageSquareText size={20} />;
}

function formatLabel(value) {
  if (!value) return "Not available";

  return String(value)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function confidencePercent(value) {
  if (typeof value !== "number") return "—";

  const percentage = value <= 1 ? value * 100 : value;

  return `${percentage.toFixed(1)}%`;
}

function urgencyClass(urgency) {
  const value = String(urgency || "").toLowerCase();

  if (value.includes("high")) return "danger";
  if (value.includes("medium")) return "warning";

  return "normal";
}

function sentimentClass(sentiment) {
  const value = String(sentiment || "").toLowerCase();

  if (value.includes("negative")) return "danger";
  if (value.includes("positive")) return "success";

  return "neutral";
}

function App() {
  const [complaint, setComplaint] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showTechnical, setShowTechnical] = useState(false);
  const [showSimilar, setShowSimilar] = useState(false);

  const analyzeComplaint = async () => {
    const text = complaint.trim();

    if (!text) {
      setError("Please enter a complaint before analyzing.");
      return;
    }

    if (text.length < 10) {
      setError("Please enter a little more detail so the AI can analyze the complaint.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);
    setShowTechnical(false);
    setShowSimilar(false);

    try {
      const response = await fetch(`${API_URL}/api/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text,
        }),
      });

      if (!response.ok) {
        let message = "The server could not analyze the complaint.";

        try {
          const data = await response.json();
          message = data.detail || message;
        } catch {
          // Keep default message.
        }

        throw new Error(message);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);

      setError(
        "Could not connect to the AI backend. Make sure the FastAPI server is running on port 8000."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      analyzeComplaint();
    }
  };

  const useSample = (sample) => {
    setComplaint(sample);
    setResult(null);
    setError("");
  };

  const resetAnalysis = () => {
    setComplaint("");
    setResult(null);
    setError("");
    setShowTechnical(false);
    setShowSimilar(false);
  };

  const issues = result?.issues || result?.detected_issues || [];
  const similarComplaints =
    result?.similar_complaints || result?.similar || [];
  const ranking = result?.ranking || result?.department_ranking || [];
  const evidence = result?.evidence || [];
  const userResponse =
    result?.user_response ||
    result?.response ||
    "The AI has completed the complaint analysis.";

  const department =
    result?.prediction ||
    result?.department ||
    result?.predicted_department ||
    "Not available";

 const sentiment =
  typeof result?.sentiment === "object"
    ? result.sentiment?.label
    : result?.sentiment || "Not available";

const urgency =
  typeof result?.urgency === "object"
    ? result.urgency?.level
    : result?.urgency || "Low";
  const confidence =
    result?.confidence ??
    result?.ai_confidence ??
    result?.probability ??
    null;

  const needsReview =
    result?.needs_review ??
    (typeof confidence === "number" && confidence < 0.5);

  return (
    <div className="app-shell">
  <header className="topbar">
    <div className="topbar-inner">

      <div className="brand">
        <img
          src="/logo.png"
          alt="Bus Complaint AI"
          className="brand-logo"
        />

        <div className="brand-text">
          <div className="brand-title">Bus Complaint AI</div>

          <div className="brand-subtitle">
            Intelligent complaint analysis
          </div>
        </div>
      </div>

      <div className="status-pill">
        <span className="status-dot" />
        AI service
      </div>

    </div>
  </header>

      <main className="page">
        <section className="hero">
          <div className="hero-badge">
            <Sparkles size={15} />
            AI-powered complaint analysis
          </div>

          <h1>
            Turn your bus complaint
            <br />
            into a clear next step.
          </h1>

          <p>
            Describe what happened. The AI will identify the likely department,
            sentiment, urgency, and important issues in your complaint.
          </p>
        </section>

        <section className="complaint-card">
          <div className="section-heading">
            <div>
              <h2>What happened?</h2>
              <p>Write your complaint in your own words.</p>
            </div>

          </div>

          <textarea
            value={complaint}
            onChange={(event) => {
              setComplaint(event.target.value);
              if (error) setError("");
            }}
            onKeyDown={handleKeyDown}
            placeholder="Example: My bus was supposed to arrive at 8 PM but it arrived two hours late..."
            maxLength={2000}
          />

          <div className="input-footer">
            <span>{complaint.length}/2000 characters</span>

            <button
              className="primary-button"
              onClick={analyzeComplaint}
              disabled={loading}
            >
              {loading ? (
                <>
                  <RefreshCw className="spin" size={18} />
                  Analyzing...
                </>
              ) : (
                <>
                  Analyze complaint
                  <ArrowRight size={18} />
                </>
              )}
            </button>
          </div>

          {error && (
            <div className="error-box">
              <XCircle size={19} />
              <span>{error}</span>
            </div>
          )}

          <div className="samples">
            <div className="samples-title">
              <Search size={15} />
              Try a sample
            </div>

            <div className="sample-list">
              {SAMPLE_COMPLAINTS.map((sample, index) => (
                <button
                  key={index}
                  className="sample-button"
                  onClick={() => useSample(sample)}
                >
                  {sample}
                </button>
              ))}
            </div>
          </div>
        </section>

        {loading && (
          <section className="loading-card">
            <div className="loading-icon">
              <Sparkles size={22} />
            </div>

            <div>
              <strong>Analyzing your complaint...</strong>
              <p>
                Checking department, sentiment, urgency, and complaint issues.
              </p>
            </div>
          </section>
        )}

        {result && !loading && (
          <section className="results-section">
            <div className="results-header">
              <div>
                <div className="result-label">
                  <CheckCircle2 size={17} />
                  Analysis complete
                </div>

                <h2>AI assessment</h2>
                <p>
                  Here is what the system found in your complaint.
                </p>
              </div>

              <button className="secondary-button" onClick={resetAnalysis}>
                Analyze another
              </button>
            </div>

            <div className="assessment-card">
              <div className="assessment-icon">
                <Sparkles size={24} />
              </div>

              <div>
                <div className="assessment-label">AI response</div>

                <p className="assessment-text">{userResponse}</p>
              </div>
            </div>

            <div className="stats-grid">
              <div className="stat-card">
                <div className="stat-top">
                  <div className="stat-icon">
                    <StatIcon type="department" />
                  </div>

                  <span>Likely department</span>
                </div>

                <strong>{formatLabel(department)}</strong>
              </div>

              <div className="stat-card">
                <div className="stat-top">
                  <div className="stat-icon">
                    {getSentimentIcon(sentiment)}
                  </div>

                  <span>Sentiment</span>
                </div>

                <strong className={sentimentClass(sentiment)}>
                  {formatLabel(sentiment)}
                </strong>
              </div>

              <div className="stat-card">
                <div className="stat-top">
                  <div className="stat-icon">
                    <StatIcon type="urgency" />
                  </div>

                  <span>Urgency</span>
                </div>

                <strong className={urgencyClass(urgency)}>
                  {formatLabel(urgency)}
                </strong>
              </div>

              <div className="stat-card">
                <div className="stat-top">
                  <div className="stat-icon">
                    <StatIcon type="confidence" />
                  </div>

                  <span>AI confidence</span>
                </div>

                <strong>{confidencePercent(confidence)}</strong>
              </div>
            </div>

            {needsReview && (
              <div className="review-warning">
                <ShieldAlert size={20} />

                <div>
                  <strong>Human review recommended</strong>
                  <p>
                    The model is less certain about this complaint. A staff
                    member should review it before taking action.
                  </p>
                </div>
              </div>
            )}

            <div className="detail-grid">
              <div className="detail-card">
                <div className="detail-card-header">
                  <div>
                    <span className="detail-eyebrow">ISSUES FOUND</span>
                    <h3>Detected issues</h3>
                  </div>

                  <MessageSquareText size={20} />
                </div>

                {issues.length > 0 ? (
                  <div className="issue-list">
                    {issues.map((issue, index) => {
                      const issueName =
                        typeof issue === "string"
                          ? issue
                          : issue.issue || issue.name || "Issue";

                      const terms =
                        typeof issue === "object"
                          ? issue.terms || []
                          : [];

                      return (
                        <div className="issue-item" key={index}>
                          <div className="issue-check">
                            <CheckCircle2 size={16} />
                          </div>

                          <div>
                            <strong>{formatLabel(issueName)}</strong>

                            {terms.length > 0 && (
                              <small>
                                Keywords: {terms.join(", ")}
                              </small>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <p className="empty-text">
                    No specific issue categories were detected.
                  </p>
                )}
              </div>

              <div className="detail-card">
                <div className="detail-card-header">
                  <div>
                    <span className="detail-eyebrow">RECOMMENDED ACTION</span>
                    <h3>Suggested next step</h3>
                  </div>

                  <ArrowRight size={20} />
                </div>

                <div className="action-box">
                  <div className="action-icon">
                    <CheckCircle2 size={19} />
                  </div>

                  <p>
                    {result?.recommended_action ||
                      result?.next_step ||
                      "Review the complaint and route it to the appropriate department."}
                  </p>
                </div>

                {(result?.urgency_reason || result?.urgency?.reason) && (
                  <div className="urgency-note">
                    <Clock3 size={17} />

                    <div>
                      <strong>Why this urgency?</strong>
                      <p>
                        {result?.urgency_reason || result?.urgency?.reason}
                        </p>
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="expandable-card">
              <button
                className="expand-button"
                onClick={() => setShowSimilar((value) => !value)}
              >
                <div className="expand-title">
                  <Search size={19} />
                  <div>
                    <strong>Similar complaints</strong>
                    <span>
                      Compare this complaint with related examples.
                    </span>
                  </div>
                </div>

                {showSimilar ? (
                  <ChevronUp size={20} />
                ) : (
                  <ChevronDown size={20} />
                )}
              </button>

              {showSimilar && (
                <div className="expand-content">
                  {similarComplaints.length > 0 ? (
                    <div className="similar-list">
                      {similarComplaints.map((item, index) => {
                        const text =
                          typeof item === "string"
                            ? item
                            : item.text ||
                              item.complaint ||
                              item.review_text ||
                              "Similar complaint";

                        const score =
                          typeof item === "object"
                            ? item.similarity ??
                              item.score ??
                              item.distance
                            : null;

                        return (
                          <div className="similar-item" key={index}>
                            <div className="similar-number">
                              {index + 1}
                            </div>

                            <div className="similar-body">
                              <p>{text}</p>

                              {score !== null && (
                                <span>
                                  Similarity:{" "}
                                  {typeof score === "number"
                                    ? score <= 1
                                      ? `${(score * 100).toFixed(1)}%`
                                      : score.toFixed(2)
                                    : score}
                                </span>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <p className="empty-text">
                      No similar complaints were returned by the model.
                    </p>
                  )}
                </div>
              )}
            </div>

            <div className="expandable-card">
              <button
                className="expand-button"
                onClick={() => setShowTechnical((value) => !value)}
              >
                <div className="expand-title">
                  <Sparkles size={19} />
                  <div>
                    <strong>Technical analysis</strong>
                    <span>
                      View model details and supporting evidence.
                    </span>
                  </div>
                </div>

                {showTechnical ? (
                  <ChevronUp size={20} />
                ) : (
                  <ChevronDown size={20} />
                )}
              </button>

              {showTechnical && (
                <div className="expand-content">
                  <div className="technical-grid">
                    <div>
                      <span className="detail-eyebrow">PROCESSED TEXT</span>

                      <div className="code-box">
                        {result?.cleaned_text ||
                          result?.processed_text ||
                          "No processed text returned."}
                      </div>
                    </div>

                    <div>
                      <span className="detail-eyebrow">
                        DEPARTMENT RANKING
                      </span>

                      {ranking.length > 0 ? (
                        <div className="ranking-list">
                          {ranking.map((item, index) => {
                            const name =
                              typeof item === "string"
                                ? item
                                : item.department ||
                                  item.label ||
                                  item.name ||
                                  "Department";

                            const score =
                              typeof item === "object"
                                ? item.confidence ??
                                  item.score ??
                                  item.probability
                                : null;

                            return (
                              <div className="ranking-row" key={index}>
                                <span>{formatLabel(name)}</span>

                                <strong>
                                  {score !== null
                                    ? confidencePercent(score)
                                    : `#${index + 1}`}
                                </strong>
                              </div>
                            );
                          })}
                        </div>
                      ) : (
                        <p className="empty-text">
                          Department ranking is not available.
                        </p>
                      )}
                    </div>
                  </div>

                  {evidence.length > 0 && (
                    <div className="evidence-block">
                      <span className="detail-eyebrow">MODEL EVIDENCE</span>

                      <div className="evidence-list">
                        {evidence.map((item, index) => (
                          <div className="evidence-item" key={index}>
                            <CheckCircle2 size={16} />
                            <span>
                              {typeof item === "string"
                                ? item
                                : JSON.stringify(item)}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="analysis-disclaimer">
              <AlertTriangle size={16} />
              <span>
                AI results are intended to assist complaint routing and should
                be reviewed by staff when confidence is low or the complaint
                involves multiple issues.
              </span>
            </div>
          </section>
        )}

        {!result && !loading && (
          <section className="how-it-works">
            <div className="section-heading centered">
              <span className="detail-eyebrow">HOW IT WORKS</span>
              <h2>From complaint to action</h2>
            </div>

            <div className="steps">
              <div className="step">
                <div className="step-number">1</div>
                <div>
                  <strong>Describe</strong>
                  <p>Tell us what happened in your own words.</p>
                </div>
              </div>

              <div className="step">
                <div className="step-number">2</div>
                <div>
                  <strong>Analyze</strong>
                  <p>The NLP model identifies the complaint category.</p>
                </div>
              </div>

              <div className="step">
                <div className="step-number">3</div>
                <div>
                  <strong>Understand</strong>
                  <p>Get sentiment, urgency, issues, and confidence.</p>
                </div>
              </div>

              <div className="step">
                <div className="step-number">4</div>
                <div>
                  <strong>Act</strong>
                  <p>Use the suggested next step to route the complaint.</p>
                </div>
              </div>
            </div>
          </section>
        )}
      </main>

      <footer className="footer">
        <div>
          <strong>Bus Complaint AI</strong>
          <span> • NLP-powered complaint analysis</span>
        </div>

        <span>Academic / demonstration system</span>
      </footer>
    </div>
  );
}

export default App;
