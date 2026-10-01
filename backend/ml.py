from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion, Pipeline


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "bus_complaints.csv"
REVIEWS_FILE = BASE_DIR / "data" / "processed" / "reviews_clean.csv"
MODEL_FILE = BASE_DIR / "artifacts" / "complaint_classifier.joblib"


# ============================================================
# DEPARTMENTS
# ============================================================

DEPARTMENTS = [
    "Operations",
    "Safety",
    "Billing",
    "Infrastructure",
    "Maintenance",
    "Customer Service",
]


# ============================================================
# SENTIMENT LEXICON
# ============================================================

NEGATIVE = {
    "angry",
    "bad",
    "delay",
    "delayed",
    "dirty",
    "late",
    "lost",
    "rude",
    "unsafe",
    "worst",
    "dangerous",
    "broken",
    "failed",
    "wrong",
    "blocked",
    "cancelled",
    "cancel",
    "missing",
    "harassed",
    "harassment",
    "accident",
    "poor",
    "horrible",
    "terrible",
    "disappointed",
    "problem",
    "issue",
    "complaint",
    "refund",
    "cheated",
    "cheater",
    "waste",
}

POSITIVE = {
    "helpful",
    "quick",
    "safe",
    "thank",
    "thanks",
    "good",
    "great",
    "excellent",
    "resolved",
    "comfortable",
    "smooth",
    "perfect",
    "friendly",
    "best",
}


# ============================================================
# DEPARTMENT HINTS
# ============================================================

DEPARTMENT_HINTS = {
    "Operations": {
        "late",
        "delay",
        "delayed",
        "timetable",
        "schedule",
        "scheduled",
        "route",
        "stop",
        "cancelled",
        "cancel",
        "departure",
        "arrival",
        "overcrowded",
        "waiting",
    },
    "Safety": {
        "dangerous",
        "unsafe",
        "accident",
        "harassed",
        "harassment",
        "emergency",
        "injury",
        "injured",
        "driver",
        "driving",
        "threat",
    },
    "Billing": {
        "charged",
        "charge",
        "fare",
        "refund",
        "refunded",
        "payment",
        "money",
        "ticket",
        "tickets",
        "card",
        "recharge",
        "debited",
        "booking",
        "price",
    },
    "Infrastructure": {
        "shelter",
        "sign",
        "ramp",
        "pavement",
        "lights",
        "bench",
        "station",
        "roof",
        "board",
        "stop",
        "road",
    },
    "Maintenance": {
        "seat",
        "seats",
        "air",
        "conditioning",
        "ac",
        "windows",
        "window",
        "engine",
        "doors",
        "door",
        "tyre",
        "tires",
        "dirty",
        "cleaned",
        "broken",
        "vehicle",
    },
    "Customer Service": {
        "helpline",
        "staff",
        "complaint",
        "website",
        "inspector",
        "email",
        "help",
        "information",
        "support",
        "customer",
        "service",
        "response",
        "contact",
    },
}


# ============================================================
# ISSUE KEYWORDS
# ============================================================

ISSUE_KEYWORDS = {
    "delay": {
        "delay",
        "delayed",
        "late",
        "lateness",
        "waiting",
        "wait",
        "timing",
        "schedule",
        "scheduled",
    },
    "refund": {
        "refund",
        "refunded",
        "money",
        "reimbursement",
        "cancelled",
        "cancellation",
        "cancel",
    },
    "booking": {
        "booking",
        "booked",
        "reservation",
        "ticket",
        "tickets",
        "pnr",
    },
    "service": {
        "service",
        "staff",
        "support",
        "customer",
        "helpline",
        "complaint",
        "response",
        "contact",
    },
    "vehicle": {
    "engine",
    "tyre",
    "tires",
    "door",
    "doors",
    "window",
    "windows",
    "seat",
    "seats",
    "ac",
    "air",
    "conditioning",
},
    "safety": {
        "unsafe",
        "dangerous",
        "accident",
        "harassment",
        "harassed",
        "emergency",
        "driver",
        "injury",
        "injured",
        "threat",
    },
    "payment": {
        "payment",
        "paid",
        "charged",
        "charge",
        "debit",
        "debited",
        "card",
        "transaction",
        "fare",
        "price",
    },
}


# ============================================================
# DEPARTMENT ACTIONS
# ============================================================

DEPARTMENT_ACTIONS = {
    "Operations": (
        "Review route, schedule, cancellation, and service-operation records."
    ),
    "Safety": (
        "Escalate to the safety team and review the reported safety incident."
    ),
    "Billing": (
        "Verify payment, fare, ticket, refund, or transaction records."
    ),
    "Infrastructure": (
        "Inspect the reported station, stop, shelter, road, or facility issue."
    ),
    "Maintenance": (
        "Inspect the vehicle or equipment and create a maintenance request."
    ),
    "Customer Service": (
        "Review the support interaction and contact the customer for resolution."
    ),
}

def build_user_response(
    department: str,
    confidence: float,
    confidence_label: str,
    sentiment_result: dict[str, Any],
    issues: list[dict[str, Any]],
    urgency: dict[str, Any],
) -> str:
    """
    Generate a simple human-readable explanation for the user.
    This is template-based so the response remains deterministic
    and grounded in the model output.
    """

    issue_names = [
        item["issue"]
        for item in issues
    ]

    readable_issues = [
        issue.replace("_", " ")
        for issue in issue_names
    ]

    if readable_issues:
        if len(readable_issues) == 1:
            issue_text = readable_issues[0]
        elif len(readable_issues) == 2:
            issue_text = (
                f"{readable_issues[0]} and "
                f"{readable_issues[1]}"
            )
        else:
            issue_text = (
                ", ".join(readable_issues[:-1])
                + f", and {readable_issues[-1]}"
            )

        response = (
            f"Your complaint appears to be mainly related "
            f"to {issue_text}."
        )
    else:
        response = (
            "The AI identified a bus-service complaint, "
            "but could not determine a specific issue with "
            "high confidence."
        )

    if len(readable_issues) > 1:
        response += (
            " Your complaint contains multiple issues, "
            "so the AI is less certain about assigning it "
            "to a single department."
        )

    if confidence < 0.40:
        response += (
            " A staff member should review the complaint "
            "before taking action."
        )

    return response


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """
    Clean complaint/review text for NLP processing.

    Removes:
    - URLs
    - email addresses
    - phone numbers
    - common ticket / PNR identifiers
    - punctuation
    - excessive whitespace
    """

    text = str(text or "").lower()

    # URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
    )

    # Email addresses
    text = re.sub(
        r"[\w.+-]+@[\w-]+\.[\w.-]+",
        " ",
        text,
    )

    # Phone numbers
    text = re.sub(
        r"\b\d{10}\b",
        " ",
        text,
    )

    # Common ticket / PNR identifiers
    text = re.sub(
    r"\b(?:pnr|tkt|ticket\s*(?:number|no|id)?)[\s:#-]*[A-Z0-9][A-Z0-9-]{4,}\b",
    " ",
    text,
    flags=re.IGNORECASE,
    )

    # Punctuation
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def tokenize(text: str) -> list[str]:
    return re.findall(
        r"[a-z0-9]+",
        clean_text(text),
    )


# ============================================================
# SENTIMENT
# ============================================================

def sentiment(text: str) -> dict[str, Any]:
    tokens = set(tokenize(text))

    negative_terms = sorted(
        tokens & NEGATIVE
    )

    positive_terms = sorted(
        tokens & POSITIVE
    )

    score = (
        len(positive_terms)
        - len(negative_terms)
    )

    if score < 0:
        label = "negative"
    elif score > 0:
        label = "positive"
    else:
        label = "neutral"

    return {
        "label": label,
        "score": score,
        "negative_terms": negative_terms,
        "positive_terms": positive_terms,
    }


# ============================================================
# MACHINE LEARNING PIPELINE
# ============================================================

def build_pipeline() -> Pipeline:

    features = FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=1,
                    max_df=0.98,
                    strip_accents="unicode",
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    sublinear_tf=True,
                    min_df=1,
                ),
            ),
        ]
    )

    return Pipeline(
        [
            (
                "features",
                features,
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


# ============================================================
# TRAINING
# ============================================================

def train_and_save() -> dict[str, Any]:

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = {
        "complaint",
        "department",
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Dataset must contain complaint and department columns."
        )

    df["cleaned"] = df["complaint"].map(
        clean_text
    )

    df = df[
        df["cleaned"].str.len() > 0
    ].copy()

    model = build_pipeline()

    model.fit(
        df["cleaned"],
        df["department"],
    )

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": model,
            "complaints": df["complaint"].tolist(),
            "departments": df["department"].tolist(),
            "cleaned": df["cleaned"].tolist(),
        },
        MODEL_FILE,
    )

    return {
        "records": int(len(df)),
        "departments": sorted(
            df["department"].unique().tolist()
        ),
        "model_file": str(MODEL_FILE),
        "status": "trained",
    }


# ============================================================
# MODEL LOADING
# ============================================================

def load_bundle() -> dict[str, Any]:

    if not MODEL_FILE.exists():
        train_and_save()

    return joblib.load(
        MODEL_FILE
    )


# ============================================================
# MODEL METADATA
# ============================================================

def model_metadata(
    bundle: dict[str, Any],
) -> dict[str, Any]:

    df = pd.DataFrame(
        {
            "complaint": bundle["complaints"],
            "department": bundle["departments"],
        }
    )

    counts = (
        df["department"]
        .value_counts()
        .reindex(DEPARTMENTS)
        .fillna(0)
        .astype(int)
    )

    return {
        "records": int(len(df)),
        "departments": DEPARTMENTS,
        "distribution": [
            {
                "department": department,
                "count": int(count),
            }
            for department, count in counts.items()
        ],
        "model": (
            "TF-IDF (word + character n-grams) "
            "+ Logistic Regression"
        ),
        "dataset_note": (
            "48 anonymized teaching examples; suitable "
            "for academic demonstration, not production deployment."
        ),
    }


# ============================================================
# ISSUE EXTRACTION
# ============================================================

def extract_issue_keywords(
    text: str,
) -> list[dict[str, Any]]:

    tokens = set(
        tokenize(text)
    )

    matches = []

    for issue, keywords in ISSUE_KEYWORDS.items():

        hits = sorted(
            tokens & keywords
        )

        if hits:
            matches.append(
                {
                    "issue": issue,
                    "terms": hits,
                    "count": len(hits),
                }
            )

    matches.sort(
        key=lambda item: item["count"],
        reverse=True,
    )

    return matches


# ============================================================
# URGENCY DETECTION
# ============================================================

def detect_urgency(
    text: str,
    sentiment_result: dict[str, Any],
) -> dict[str, Any]:

    tokens = set(tokenize(text))

    # ========================================================
    # HIGH URGENCY
    # Immediate safety, medical, fire, or serious threat
    # ========================================================
    high_terms = {
        "accident",
        "emergency",
        "dangerous",
        "unsafe",
        "injury",
        "injured",
        "fire",
        "threat",
        "harassment",
        "harassed",
        "assault",
        "collision",
        "crash",
        "smoke",
        "medical",
        "hospital",
        "bleeding",
        "hurt",
        "violence",
    }

    # ========================================================
    # MEDIUM URGENCY
    # Significant service disruption requiring attention
    # ========================================================
    medium_terms = {
        "stranded",
        "breakdown",
        "broken",
        "cancelled",
        "canceled",
        "stuck",
        "missed",
        "delay",
        "delayed",
        "late",
        "hours",
        "waiting",
        "replacement",
        "rerouted",
        "diverted",
        "no-show",
        "unavailable",
        "blocked",
    }

    high_hits = sorted(tokens & high_terms)
    medium_hits = sorted(tokens & medium_terms)

    # ========================================================
    # HIGH HAS PRIORITY
    # ========================================================
    if high_hits:
        return {
            "level": "high",
            "reason": (
                "High-priority safety or emergency indicators "
                "were detected."
            ),
            "terms": high_hits,
        }

    # ========================================================
    # MEDIUM
    # ========================================================
    if medium_hits:
        return {
            "level": "medium",
            "reason": (
                "The complaint contains significant service "
                "disruption indicators that may require prompt attention."
            ),
            "terms": medium_hits,
        }

    # ========================================================
    # LOW
    # ========================================================
    return {
        "level": "low",
        "reason": (
            "No high-priority safety or significant service "
            "disruption indicators were detected."
        ),
        "terms": [],
    }

# ============================================================
# EXPLAINABLE EVIDENCE
# ============================================================

def extract_department_evidence(
    text: str,
) -> list[dict[str, Any]]:

    tokens = set(
        tokenize(text)
    )

    evidence = []

    for department, hints in DEPARTMENT_HINTS.items():

        hits = sorted(
            tokens & hints
        )

        if hits:

            evidence.append(
                {
                    "department": department,
                    "terms": hits,
                    "count": len(hits),
                }
            )

    evidence.sort(
        key=lambda item: item["count"],
        reverse=True,
    )

    return evidence


# ============================================================
# SIMILAR COMPLAINTS
# ============================================================

def find_similar_complaints(
    cleaned_text: str,
    bundle: dict[str, Any],
    top_k: int = 3,
) -> list[dict[str, Any]]:

    if not cleaned_text:
        return []

    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    corpus = word_vectorizer.fit_transform(
        bundle["cleaned"]
    )

    query = word_vectorizer.transform(
        [cleaned_text]
    )

    similarities = cosine_similarity(
        query,
        corpus,
    )[0]

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    similar = []

    for index in top_indices:

        index = int(index)

        similar.append(
            {
                "complaint": bundle["complaints"][index],
                "department": bundle["departments"][index],
                "similarity": round(
                    float(similarities[index]),
                    4,
                ),
            }
        )

    return similar


# ============================================================
# MAIN PREDICTION
# ============================================================

def predict(
    text: str,
) -> dict[str, Any]:

    if not text or not text.strip():
        raise ValueError(
            "Complaint text cannot be empty."
        )

    bundle = load_bundle()

    model: Pipeline = bundle["model"]

    cleaned = clean_text(
        text
    )

    if not cleaned:
        raise ValueError(
            "Complaint does not contain usable text."
        )

    # --------------------------------------------
    # Department prediction
    # --------------------------------------------

    probabilities = model.predict_proba(
        [cleaned]
    )[0]

    classes = model.classes_

    ranking = sorted(
        [
            {
                "department": str(department),
                "probability": round(
                    float(probability),
                    4,
                ),
            }
            for department, probability
            in zip(classes, probabilities)
        ],
        key=lambda item: item["probability"],
        reverse=True,
    )

    predicted = ranking[0]

    confidence = float(
        predicted["probability"]
    )

    # --------------------------------------------
    # Confidence label
    # --------------------------------------------

    if confidence >= 0.60:
        confidence_label = "high"
    elif confidence >= 0.40:
        confidence_label = "medium"
    else:
        confidence_label = "low"

    needs_review = confidence < 0.40

    # --------------------------------------------
    # Sentiment
    # --------------------------------------------

    sentiment_result = sentiment(
        text
    )

    # --------------------------------------------
    # Issue extraction
    # --------------------------------------------

    issues = extract_issue_keywords(
        text
    )

    # --------------------------------------------
    # Urgency
    # --------------------------------------------

    urgency = detect_urgency(
        text,
        sentiment_result,
    )

    # --------------------------------------------
    # Explainable evidence
    # --------------------------------------------

    evidence = extract_department_evidence(
        text
    )

    # --------------------------------------------
    # Similar complaints
    # --------------------------------------------

    similar = find_similar_complaints(
        cleaned,
        bundle,
        top_k=3,
    )

    # --------------------------------------------
    # Recommended action
    # --------------------------------------------

    department = str(
        predicted["department"]
    )

    recommended_action = DEPARTMENT_ACTIONS.get(
        department,
        "Review the complaint manually.",
    )
    user_response = build_user_response(
    department=department,
    confidence=confidence,
    confidence_label=confidence_label,
    sentiment_result=sentiment_result,
    issues=issues,
    urgency=urgency,
    )

    # --------------------------------------------
    # Final response
    # --------------------------------------------

    return {
        "complaint": text.strip(),

        "user_response": user_response,

        "cleaned_text": cleaned,

        "prediction": department,

        "confidence": round(
            confidence,
            4,
        ),

        "confidence_label": confidence_label,

        "needs_review": needs_review,

        "ranking": ranking,

        "sentiment": sentiment_result,

        "issues": issues,

        "urgency": urgency,

        "recommended_action": recommended_action,

        "evidence": evidence,

        "similar_complaints": similar,

        "tokens": tokenize(text),
    }


# ============================================================
# REVIEW DATASET ANALYTICS
# ============================================================

def review_analytics() -> dict[str, Any]:

    if not REVIEWS_FILE.exists():

        return {
            "available": False,
            "message": (
                "Processed reviews dataset not found."
            ),
        }

    df = pd.read_csv(
        REVIEWS_FILE
    )

    if df.empty:

        return {
            "available": False,
            "message": "Review dataset is empty.",
        }

    # --------------------------------------------
    # Rating distribution
    # --------------------------------------------

    rating_counts = (
        df["rating"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    # --------------------------------------------
    # Sentiment distribution
    # --------------------------------------------

    sentiment_counts = (
        df["sentiment"]
        .value_counts()
        .to_dict()
    )

    # --------------------------------------------
    # Useful top terms
    # --------------------------------------------

    words = (
        " ".join(
            df["cleaned_text"]
            .fillna("")
            .astype(str)
        )
        .split()
    )

    stop_words = {
        "the",
        "and",
        "to",
        "a",
        "of",
        "in",
        "is",
        "for",
        "on",
        "with",
        "this",
        "that",
        "it",
        "was",
        "are",
        "i",
        "we",
        "you",
        "they",
        "my",
        "very",
        "have",
        "had",
        "but",
        "not",
        "from",
        "be",
        "as",
        "at",
        "our",
        "your",
        "so",
        "were",
        "has",
        "or",
        "an",
        "me",
        "no",
        "by",
        "t",
        "red",
        "redbus",
    
    }

    filtered_words = [
        word
        for word in words
        if word not in stop_words
        and len(word) >= 3
    ]

    if filtered_words:

        word_counts = (
            pd.Series(filtered_words)
            .value_counts()
            .head(15)
        )

    else:

        word_counts = pd.Series(
            dtype=int
        )

    # --------------------------------------------
    # Review length
    # --------------------------------------------

    review_lengths = (
        df["cleaned_text"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    return {
        "available": True,

        "review_count": int(
            len(df)
        ),

        "average_rating": round(
            float(
                df["rating"].mean()
            ),
            2,
        ),

        "rating_distribution": [
            {
                "rating": int(rating),
                "count": int(count),
            }
            for rating, count
            in rating_counts.items()
        ],

        "sentiment_distribution": [
            {
                "sentiment": str(label),
                "count": int(count),
            }
            for label, count
            in sentiment_counts.items()
        ],

        "top_terms": [
            {
                "term": str(term),
                "count": int(count),
            }
            for term, count
            in word_counts.items()
        ],

        "review_length": {
            "average_words": round(
                float(
                    review_lengths.mean()
                ),
                2,
            ),
            "median_words": round(
                float(
                    review_lengths.median()
                ),
                2,
            ),
            "maximum_words": int(
                review_lengths.max()
            ),
        },
    }


# ============================================================
# COMPLETE ANALYTICS
# ============================================================

def analytics() -> dict[str, Any]:

    bundle = load_bundle()

    metadata = model_metadata(
        bundle
    )

    metadata["reviews"] = review_analytics()

    return metadata