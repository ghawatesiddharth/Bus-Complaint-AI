# TransitIQ — Bus Complaint AI

A professional NLP project that classifies passenger complaints into the department responsible for handling them.

## What this version adds

- FastAPI inference API
- React + Vite dashboard
- Saved `joblib` model artifact
- Word + character TF-IDF features
- Logistic Regression classifier
- Confidence ranking with low-confidence human-review flag
- Sentiment scoring
- Explainable keyword evidence
- Similar historical complaint retrieval
- Dataset distribution dashboard
- Reproducible training script
- Clear limitation disclosure for the 48-row academic dataset

## Departments

Operations · Safety · Billing · Infrastructure · Maintenance · Customer Service

## Architecture

```text
React / Vite UI
       |
       | POST /api/predict
       v
FastAPI
       |
       v
Text normalization
       |
       +--> TF-IDF word n-grams ----\
       |                              > Logistic Regression
       +--> TF-IDF character n-grams/
       |
       +--> sentiment + lexical evidence
       |
       +--> similarity retrieval
       |
       v
Prediction + confidence + explanation
```

## Run backend

From the repository root:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r backend/requirements.txt
python -m backend.train
uvicorn backend.app:app --reload --port 8000
```

API docs: http://127.0.0.1:8000/docs

## Run frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal, normally http://localhost:5173.

## Test examples

- `The ticket machine charged my card twice for one journey` → Billing
- `The driver was using a phone and driving dangerously` → Safety
- `The bus arrived forty minutes late and I missed my class` → Operations
- `The bus stop shelter is broken and needs repair` → Infrastructure

## Important evaluation note

The supplied dataset contains only 48 labeled teaching examples, eight per department. A train/test split of the original implementation produced 75% accuracy and 66.67% for its Naive Bayes baseline. Those numbers are useful for demonstrating the workflow but are not evidence of production performance.

For a serious deployment, collect substantially more labeled complaints, establish annotation guidelines, use cross-validation, track macro-F1, and evaluate on a held-out time-based test set.

## Project structure

```text
NLP-main-improved/
├── backend/
│   ├── app.py
│   ├── ml.py
│   ├── train.py
│   ├── requirements.txt
│   └── data/
│       └── bus_complaints.csv
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── main.jsx
│       └── styles.css
└── README.md
```
