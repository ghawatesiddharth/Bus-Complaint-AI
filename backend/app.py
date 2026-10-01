from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .ml import (
    analytics,
    load_bundle,
    predict,
    train_and_save,
)


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Bus Complaint AI",
    version="2.0.0",
    description=(
        "NLP-powered bus complaint classification, "
        "sentiment analysis, urgency detection, "
        "similar complaint retrieval, and analytics API."
    ),
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODELS
# ============================================================

class ComplaintRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Bus complaint text.",
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    try:

        bundle = load_bundle()

        return {
            "status": "ok",
            "service": "bus-complaint-ai",
            "model_loaded": bundle is not None,
            "version": "2.0.0",
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="ML model could not be loaded.",
        ) from exc


# ============================================================
# ANALYTICS
# ============================================================

@app.get("/api/analytics")
def get_analytics():

    try:

        return analytics()

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Analytics could not be generated.",
        ) from exc


# ============================================================
# PREDICTION
# ============================================================

@app.post("/api/predict")
def classify(
    request: ComplaintRequest,
):

    try:

        return predict(
            request.text
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc


# ============================================================
# RETRAIN
# ============================================================

@app.post("/api/retrain")
def retrain():

    try:

        return train_and_save()

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Model retraining failed.",
        ) from exc