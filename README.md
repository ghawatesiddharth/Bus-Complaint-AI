# TransitIQ — Bus Complaint AI

> **An explainable NLP system for automatically routing passenger complaints to the department responsible for handling them.**

TransitIQ is a B-Tech academic NLP project that combines a FastAPI backend, a React + Vite frontend, TF-IDF text representations, and a Logistic Regression classifier to analyze bus passenger complaints.

The application predicts the responsible department, estimates confidence, detects sentiment and urgency, highlights keyword evidence, and retrieves similar historical complaints.

---

## 🚀 Live Demo

### Frontend

**TransitIQ Web App:**  
https://bus-complaint-ai.onrender.com

### Backend API

**FastAPI API:**  
https://bus-complaint-ai-api.onrender.com/api/health

### API Documentation

**Swagger UI:**  
https://bus-complaint-ai-api.onrender.com/docs

> The backend is deployed on Render's free web-service tier. Free services can spin down after a period of inactivity, so the first request after inactivity may take longer than usual.

---

## 🎯 What TransitIQ Does

A passenger enters a complaint such as:

> "The ticket machine charged my card twice for one journey."

TransitIQ processes the text and returns information such as:

- **Department:** Billing
- **Sentiment:** Negative
- **Urgency:** Low / Medium / High
- **Confidence:** Model confidence score
- **Detected issues:** Relevant complaint keywords
- **Recommended action:** Plain-language handling guidance
- **Technical analysis:** Model evidence and prediction details
- **Similar complaints:** Related historical complaints

---

## ✨ Key Features

- FastAPI inference API
- React + Vite web dashboard
- Saved `joblib` model artifact
- Word-level TF-IDF n-gram features
- Character-level TF-IDF n-gram features
- Logistic Regression classifier
- Class-balanced training
- Confidence ranking
- Low-confidence human-review flag
- Sentiment analysis
- Rule-based urgency detection
- Explainable keyword evidence
- Similar historical complaint retrieval
- Dataset distribution analytics
- Reproducible training script
- Swagger/OpenAPI API documentation
- Responsive web interface
- Deployment-ready backend and frontend
- Clear academic-dataset limitation disclosure

---

## 🏢 Departments

TransitIQ currently classifies complaints into six departments:

| Department | Typical Complaint Area |
|---|---|
| **Operations** | Delays, schedules, route/service problems |
| **Safety** | Dangerous driving, unsafe behavior, security concerns |
| **Billing** | Payments, fares, duplicate charges, ticketing |
| **Infrastructure** | Bus stops, shelters, stations and facilities |
| **Maintenance** | Vehicle condition, broken equipment and repairs |
| **Customer Service** | Staff interaction, support and passenger service |

---

# 🏗️ System Architecture

```text
                         ┌─────────────────────┐
                         │       Passenger     │
                         │    Complaint Text   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   React + Vite UI   │
                         │  Complaint Dashboard│
                         └──────────┬──────────┘
                                    │
                              POST /api/predict
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    FastAPI Backend  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Text Normalization  │
                         └──────────┬──────────┘
                                    │
                   ┌────────────────┼────────────────┐
                   │                │                │
                   ▼                ▼                ▼
          ┌────────────────┐ ┌───────────────┐ ┌──────────────────┐
          │ Word TF-IDF    │ │ Character     │ │ Sentiment +      │
          │ n-grams        │ │ TF-IDF n-grams│ │ Urgency Evidence │
          └───────┬────────┘ └───────┬───────┘ └────────┬─────────┘
                  │                  │                  │
                  └──────────┬───────┘                  │
                             ▼                          │
                   ┌─────────────────────┐              │
                   │ Logistic Regression │              │
                   │   Classifier        │              │
                   └──────────┬──────────┘              │
                              │                         │
                              ▼                         │
                   ┌─────────────────────┐              │
                   │ Department +        │◄─────────────┘
                   │ Confidence Ranking  │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ Similar Complaint   │
                   │ Retrieval           │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ Structured API      │
                   │ Response            │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ Human-Friendly UI   │
                   └─────────────────────┘
```

---

# 🤖 Machine Learning Pipeline

TransitIQ uses a combined text-feature approach.

## 1. Text Normalization

Complaint text is normalized before feature extraction.

## 2. Word TF-IDF

Word-level n-grams capture meaningful phrases and vocabulary.

Examples:

```text
"charged twice"
"bus was late"
"dangerous driving"
```

## 3. Character TF-IDF

Character-level n-grams provide additional robustness for spelling variations, partial words, and short textual patterns.

## 4. Feature Combination

The word and character representations are combined into a single feature representation.

## 5. Logistic Regression

The combined features are passed to a Logistic Regression classifier with balanced class weights.

## 6. Confidence Ranking

The classifier probabilities are used to rank department predictions and identify low-confidence cases that may require human review.

---

# 📊 Sentiment and Urgency

TransitIQ supplements the department classifier with additional complaint analysis.

## Sentiment

The project uses the following simple sentiment mapping for the review dataset:

```text
Rating <= 2  → Negative
Rating == 3  → Neutral
Rating >= 4  → Positive
```

## Urgency

Urgency is detected using lexical rules.

### HIGH

```text
accident
emergency
dangerous
unsafe
injury
injured
fire
threat
harassment
harassed
assault
collision
crash
smoke
medical
hospital
bleeding
hurt
violence
```

### MEDIUM

```text
stranded
breakdown
broken
cancelled
canceled
stuck
missed
delay
delayed
late
hours
waiting
replacement
rerouted
diverted
no-show
unavailable
blocked
```

### LOW

Complaints without high- or medium-urgency indicators.

> Urgency detection is intentionally simple and is suitable for the current academic/demo scope.

---

# 🔍 Explainability

TransitIQ does not return only a department label.

The interface also exposes supporting information such as:

- Detected complaint keywords
- Model confidence
- Ranked department probabilities
- Sentiment
- Urgency
- Similar historical complaints
- Recommended action

This makes the result easier to inspect during demonstrations and academic evaluation.

---

# 🧪 Test Examples

Try these complaints in the web interface:

| Complaint | Expected Department |
|---|---|
| `The ticket machine charged my card twice for one journey.` | Billing |
| `The driver was using a phone and driving dangerously.` | Safety |
| `The bus arrived forty minutes late and I missed my class.` | Operations |
| `The bus stop shelter is broken and needs repair.` | Infrastructure |

Additional examples:

```text
The bus AC is not working and the seats are damaged.
```

```text
The conductor was rude when I asked for help.
```

---

# 📁 Project Structure

```text
Bus_complaint/
│
├── .gitignore
├── README.md
├── BUS_COMPLAINT_ANALYSER.md
├── verify_project.py
│
├── backend/
│   ├── __init__.py
│   ├── app.py
│   ├── ml.py
│   ├── train.py
│   ├── requirements.txt
│   │
│   ├── artifacts/
│   │   └── complaint_classifier.joblib
│   │
│   └── data/
│       ├── bus_complaints.csv
│       ├── clean_reviews.py
│       ├── reviews.csv
│       └── processed/
│           └── reviews_clean.csv
│
└── frontend/
    ├── index.html
    ├── package.json
    ├── package-lock.json
    ├── vite.config.js
    │
    ├── public/
    │   └── logo.png
    │
    └── src/
        ├── App.jsx
        ├── main.jsx
        └── styles.css
```

### Important Dataset Note

The raw:

```text
backend/data/reviews.csv
```

file contains identifying information and is intentionally excluded from Git tracking.

The cleaned dataset is used for the repository/demo workflow.

---

# 💻 Local Development

## Prerequisites

Install:

- Python 3
- Node.js
- npm
- Git

---

# 1. Clone the Repository

```bash
git clone https://github.com/ghawatesiddharth/Bus-Complaint-AI.git
cd Bus-Complaint-AI
```

---

# 2. Set Up the Backend

Create and activate a Python virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install backend dependencies:

```bash
pip install -r backend/requirements.txt
```

---

# 3. Train the Model

The repository includes a reproducible training script.

From the project root:

```bash
python -m backend.train
```

The trained model artifact is written to:

```text
backend/artifacts/complaint_classifier.joblib
```

---

# 4. Start the FastAPI Backend

From the repository root:

```bash
uvicorn backend.app:app --reload --port 8000
```

Backend:

```text
http://127.0.0.1:8000
```

Swagger API documentation:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/api/health
```

---

# 5. Start the React Frontend

Open a second terminal.

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal, normally:

```text
http://localhost:5173
```

---

# 🔌 API Endpoints

## `GET /api/health`

Checks whether the API is running and whether the model is loaded.

Example response:

```json
{
  "status": "ok",
  "service": "bus-complaint-ai",
  "model_loaded": true,
  "version": "2.0.0"
}
```

---

## `POST /api/predict`

Analyzes a passenger complaint.

Example request:

```json
{
  "text": "The bus arrived forty minutes late and I missed my class."
}
```

The response contains prediction and analysis information used by the frontend.

---

## `GET /api/analytics`

Returns dataset-level analytics used by the dashboard.

---

## `POST /api/retrain`

Triggers model retraining using the configured training data.

> Use this endpoint carefully in a deployed environment. Retraining is intended for the academic/demo workflow and should be protected or restricted in a production system.

---

# ☁️ Deployment

TransitIQ is structured as two deployable services.

```text
                         GitHub Repository
                                │
                ┌───────────────┴───────────────┐
                │                               │
                ▼                               ▼
       Render Web Service              Render Static Site
       FastAPI Backend                 React + Vite Frontend
                │                               │
                │                               │
                └───────────────┬───────────────┘
                                │
                                ▼
                         TransitIQ System
```

## Backend Deployment

Backend build command:

```text
pip install -r backend/requirements.txt
```

Backend start command:

```text
uvicorn backend.app:app --host 0.0.0.0 --port $PORT
```

## Frontend Deployment

Frontend root directory:

```text
frontend
```

Build command:

```text
npm install && npm run build
```

Publish directory:

```text
dist
```

The production frontend is configured to communicate with the deployed FastAPI backend.

---

# 📈 Dataset and Evaluation

The supplied department-classification dataset contains:

- **48 labeled complaint examples**
- **6 departments**
- **8 examples per department**

Because the dataset is very small, evaluation results should be treated as academic demonstrations rather than evidence of production performance.

The original implementation reported:

- **75% accuracy** for its train/test split
- **66.67% accuracy** for its Naive Bayes baseline

These figures are retained as historical project context.

They should **not** be interpreted as a reliable estimate of real-world model performance.

For a serious deployment, the project should use:

1. A substantially larger labeled complaint dataset
2. Clear annotation guidelines
3. Stratified cross-validation
4. Macro-F1
5. Per-class precision and recall
6. A genuinely held-out test set
7. Preferably a time-based evaluation split
8. Error analysis by department
9. Monitoring for data and concept drift

---

# ⚠️ Limitations

TransitIQ is an **academic NLP project**, not a production-grade transit decision system.

Important limitations include:

- Very small labeled classification dataset
- Limited linguistic diversity
- Potential overfitting
- Limited representation of real passenger complaints
- Rule-based urgency detection
- Simple sentiment methodology
- Similarity retrieval depends on available historical examples
- Model confidence is not the same as real-world correctness
- No human annotation workflow is currently integrated
- No production authentication/authorization layer
- No enterprise-scale monitoring or model governance

The system should therefore be used for:

- Demonstration
- Learning
- Experimentation
- Academic evaluation

---

# 🔮 Future Improvements

Potential next steps include:

- Expand the labeled complaint dataset
- Add multilingual complaint support
- Add Marathi/Hindi/English language detection
- Improve sentiment classification
- Train a dedicated urgency classifier
- Compare Logistic Regression with SVM
- Compare against Naive Bayes
- Experiment with transformer-based models
- Add cross-validation
- Add automated evaluation
- Add confusion matrices
- Add per-class metrics
- Add human-review feedback
- Add model versioning
- Add database-backed complaint history
- Add authentication
- Add role-based access
- Add production monitoring
- Add automated retraining pipelines
- Add explainable model visualizations
- Introduce transformer embeddings for semantic similarity
- Improve duplicate and near-duplicate detection

---

# 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React |
| Build Tool | Vite |
| Styling | CSS |
| Backend | FastAPI |
| Server | Uvicorn |
| Machine Learning | scikit-learn |
| Feature Extraction | Word + Character TF-IDF |
| Classifier | Logistic Regression |
| Model Storage | Joblib |
| Data Processing | pandas / NumPy |
| Similarity | TF-IDF-based similarity |
| API Documentation | Swagger / OpenAPI |
| Deployment | Render |
| Version Control | Git + GitHub |

---

# 🔁 Reproducibility

The main model-training workflow is included in the repository.

Clone the project:

```bash
git clone https://github.com/ghawatesiddharth/Bus-Complaint-AI.git
cd Bus-Complaint-AI
```

Create the environment:

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r backend/requirements.txt
```

Train the model:

```bash
python -m backend.train
```

Run the backend:

```bash
uvicorn backend.app:app --reload --port 8000
```

Then open another terminal and run:

```bash
cd frontend
npm install
npm run dev
```

---

# 🎓 Academic Project Note

TransitIQ was developed as a B-Tech academic project to demonstrate an end-to-end Natural Language Processing workflow.

The complete workflow can be summarized as:

```text
Raw Complaint
      ↓
Text Preprocessing
      ↓
Feature Extraction
      ↓
Machine Learning Classification
      ↓
Confidence + Explanation
      ↓
Complaint Routing
      ↓
User Interface
```

The project demonstrates how NLP and machine learning can support the organization and routing of passenger complaints.

---

# 📌 Project Goals

The main goals of TransitIQ are to demonstrate:

- Natural Language Processing
- Text classification
- TF-IDF feature engineering
- Machine learning model training
- Explainable predictions
- Sentiment analysis
- Urgency detection
- Similarity-based retrieval
- REST API development
- React frontend development
- Full-stack ML application architecture
- Cloud deployment

---

# 📜 License

No open-source license has been specified for this repository.

Unless a license is added to the repository, the default copyright rules apply to the original project code and content.

---

# 👨‍💻 Project

**TransitIQ — Bus Complaint AI**

Built as an academic B-Tech Natural Language Processing project.

**GitHub:**  
https://github.com/ghawatesiddharth/Bus-Complaint-AI

**Live Application:**  
https://bus-complaint-ai.onrender.com

**API:**  
https://bus-complaint-ai-api.onrender.com
