# 🚌 TransitIQ — Bus Complaint AI

> An explainable NLP system that automatically routes passenger complaints to the department responsible for handling them.

TransitIQ is a B-Tech academic Natural Language Processing project that combines a **FastAPI** backend, a **React + Vite** frontend, **TF-IDF** text representations, and a **Logistic Regression** classifier to analyze bus passenger complaints.

The system predicts the responsible department, estimates model confidence, detects sentiment and urgency, identifies supporting keywords, and retrieves similar historical complaints.

---

## 📑 Table of Contents

- [Live Demo](#-live-demo)
- [What is TransitIQ?](#-what-is-transitiq)
- [Key Features](#-key-features)
- [Complaint Departments](#-complaint-departments)
- [System Architecture](#️-system-architecture)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Sentiment Analysis](#-sentiment-analysis)
- [Urgency Detection](#-urgency-detection)
- [Explainability](#-explainability)
- [Test Examples](#-test-examples)
- [Project Structure](#-project-structure)
- [Local Development](#-local-development)
- [API Endpoints](#-api-endpoints)
- [Deployment](#️-deployment)
- [Dataset and Evaluation](#-dataset-and-evaluation)
- [Limitations](#️-limitations)
- [Future Improvements](#-future-improvements)
- [Technology Stack](#️-technology-stack)
- [Reproducibility](#-reproducibility)
- [Academic Project Note](#-academic-project-note)
- [License](#-license)

---

## 🚀 Live Demo

| Component | Link |
|---|---|
| 🌐 **Frontend (Vercel)** | https://bus-complaint-ai.vercel.app |
| 🤖 **Backend API (Render)** | https://bus-complaint-ai-api.onrender.com |
| ❤️ **Health Check** | https://bus-complaint-ai-api.onrender.com/api/health |
| 📚 **API Docs (Swagger / OpenAPI)** | https://bus-complaint-ai-api.onrender.com/docs |

> **Note:** The frontend is deployed on Vercel and the FastAPI backend is deployed on Render. The Render free web-service tier may spin down after inactivity, so the first API request after a period of inactivity can take longer while the backend starts again.

---

## 🎯 What is TransitIQ?

TransitIQ is an intelligent bus complaint analysis system designed to help organize and route passenger complaints.

A passenger enters a complaint such as:

> *"The ticket machine charged my card twice for one journey."*

The system analyzes the complaint and returns:

- **Department**
- **Sentiment**
- **Urgency**
- **Confidence**
- **Detected issues**
- **Recommended action**
- **Technical analysis**
- **Similar historical complaints**

The goal is to demonstrate how Natural Language Processing and Machine Learning can support complaint classification and routing.

---

## ✨ Key Features

- FastAPI inference API
- React + Vite frontend
- Word-level TF-IDF features
- Character-level TF-IDF features
- Logistic Regression classifier with balanced class weights
- Saved Joblib model artifact
- Department classification with confidence ranking
- Low-confidence human-review flag
- Sentiment analysis
- Rule-based urgency detection
- Explainable keyword evidence
- Similar historical complaint retrieval
- Dataset distribution analytics
- Reproducible model training
- Swagger / OpenAPI documentation
- Responsive web interface
- Vercel frontend deployment and Render backend deployment
- Academic dataset limitation disclosure

---

## 🏢 Complaint Departments

TransitIQ currently classifies complaints into **six departments**.

| Department | Typical Complaint Area |
|---|---|
| **Operations** | Delays, schedules, routes and service issues |
| **Safety** | Dangerous driving, unsafe behavior and security concerns |
| **Billing** | Payments, fares, duplicate charges and ticketing |
| **Infrastructure** | Bus stops, shelters, stations and facilities |
| **Maintenance** | Vehicle condition, broken equipment and repairs |
| **Customer Service** | Staff interaction, support and passenger service |

---

## 🏗️ System Architecture

```text
                         ┌─────────────────────┐
                         │      Passenger      │
                         │   Complaint Text    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   React + Vite UI   │
                         │ Complaint Dashboard │
                         └──────────┬──────────┘
                                    │
                             POST /api/predict
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   FastAPI Backend   │
                         │      on Render      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Text Normalization  │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
          ┌────────────────┐ ┌───────────────┐ ┌──────────────────┐
          │   Word TF-IDF  │ │   Character   │ │  Sentiment +     │
          │    n-grams     │ │ TF-IDF n-grams│ │ Urgency Evidence │
          └───────┬────────┘ └───────┬───────┘ └────────┬─────────┘
                  │                  │                  │
                  └──────────┬───────┘                  │
                             ▼                          │
                   ┌─────────────────────┐              │
                   │ Logistic Regression │              │
                   │     Classifier      │              │
                   └──────────┬──────────┘              │
                              │                         │
                              ▼                         │
                   ┌─────────────────────┐              │
                   │    Department +     │◄─────────────┘
                   │ Confidence Ranking  │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │  Similar Complaint  │
                   │      Retrieval      │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │   Structured API    │
                   │      Response       │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │  Human-Friendly UI  │
                   └─────────────────────┘
```

---

## 🤖 Machine Learning Pipeline

TransitIQ uses a combined text-feature approach.

### 1. Text Normalization
Complaint text is normalized before feature extraction.

### 2. Word TF-IDF
Word-level n-grams capture meaningful vocabulary and phrases, for example:

- `"charged twice"`
- `"bus was late"`
- `"dangerous driving"`

### 3. Character TF-IDF
Character-level n-grams provide additional robustness for spelling variations, partial words and short textual patterns.

### 4. Feature Combination
The word-level and character-level TF-IDF representations are combined into a single feature representation.

### 5. Logistic Regression
The combined representation is passed to a Logistic Regression classifier using balanced class weights.

### 6. Confidence Ranking
Classifier probabilities are used to rank department predictions and to flag low-confidence cases that may require human review.

---

## 📊 Sentiment Analysis

The project uses a simple rating-based sentiment mapping for the review dataset:

```text
Rating <= 2  →  Negative
Rating == 3  →  Neutral
Rating >= 4  →  Positive
```

This sentiment component is intended for the current academic/demo scope.

---

## 🚨 Urgency Detection

TransitIQ includes a rule-based urgency detection system with three levels.

### 🔴 HIGH
Examples of high-urgency indicators:

`accident` · `emergency` · `dangerous` · `unsafe` · `injury` · `injured` · `fire` · `threat` · `harassment` · `harassed` · `assault` · `collision` · `crash` · `smoke` · `medical` · `hospital` · `bleeding` · `hurt` · `violence`

### 🟠 MEDIUM
Examples of medium-urgency indicators:

`stranded` · `breakdown` · `broken` · `cancelled` · `canceled` · `stuck` · `missed` · `delay` · `delayed` · `late` · `hours` · `waiting` · `replacement` · `rerouted` · `diverted` · `no-show` · `unavailable` · `blocked`

### 🟢 LOW
Complaints without high- or medium-urgency indicators are classified as low urgency.

> Urgency detection is intentionally simple and is suitable for the current academic/demo scope.

---

## 🔍 Explainability

TransitIQ does not return only a department label. The interface provides supporting information including:

- Detected complaint keywords
- Model confidence
- Ranked department probabilities
- Sentiment
- Urgency
- Similar historical complaints
- Recommended action

This makes the system easier to inspect and demonstrate during academic evaluation.

---

## 🧪 Test Examples

Try the following complaints in the application.

| Complaint | Expected Department |
|---|---|
| The ticket machine charged my card twice for one journey. | Billing |
| The driver was using a phone and driving dangerously. | Safety |
| The bus arrived forty minutes late and I missed my class. | Operations |
| The bus stop shelter is broken and needs repair. | Infrastructure |

Additional examples:

- The bus AC is not working and the seats are damaged.
- The conductor was rude when I asked for help.

---

## 📁 Project Structure

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

### 🔒 Dataset Privacy Note

The raw file `backend/data/reviews.csv` contains identifying information and is **intentionally excluded from Git tracking**. The cleaned dataset is used for the repository and demo workflow.

---

## 💻 Local Development

### Prerequisites

- Python 3
- Node.js
- npm
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/ghawatesiddharth/Bus-Complaint-AI.git
cd Bus-Complaint-AI
```

### 2. Set Up the Backend

Create a Python virtual environment.

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r backend/requirements.txt
```

### 3. Train the Model

From the project root:

```bash
python -m backend.train
```

The trained model artifact is generated at `backend/artifacts/complaint_classifier.joblib`.

### 4. Start the FastAPI Backend

From the project root:

```bash
uvicorn backend.app:app --reload --port 8000
```

| Service | URL |
|---|---|
| Local backend | http://127.0.0.1:8000 |
| Swagger API docs | http://127.0.0.1:8000/docs |
| Health check | http://127.0.0.1:8000/api/health |

### 5. Start the React Frontend

Open a **second terminal**:

```bash
cd frontend
npm install
npm run dev
```

The Vite development server normally runs at http://localhost:5173.

---

## 🔌 API Endpoints

### `GET /api/health`

Checks whether the API is running and whether the model is loaded.

```json
{
  "status": "ok",
  "service": "bus-complaint-ai",
  "model_loaded": true,
  "version": "2.0.0"
}
```

### `POST /api/predict`

Analyzes a passenger complaint.

**Example request:**

```json
{
  "text": "The bus arrived forty minutes late and I missed my class."
}
```

The response contains the prediction and analysis information used by the frontend.

### `GET /api/analytics`

Returns dataset-level analytics used by the dashboard.

### `POST /api/retrain`

Triggers model retraining using the configured training data.

> ⚠️ Use this endpoint carefully in a deployed environment. Retraining is intended for the academic/demo workflow and should be protected or restricted in a production system.

---

## ☁️ Deployment

TransitIQ uses separate hosting for the frontend and backend.

```text
                         GitHub Repository
                                │
                ┌───────────────┴───────────────┐
                │                               │
                ▼                               ▼
             Vercel                           Render
        React + Vite Frontend             FastAPI Backend
                │                               │
                │         API Requests          │
                └──────────────────────────────►│
                                                │
                                                ▼
                                            ML Model
```

### 🌐 Frontend — Vercel

| Setting | Value |
|---|---|
| Root Directory | `frontend` |
| Build Command | `npm run build` |
| Output Directory | `dist` |
| Production URL | https://bus-complaint-ai.vercel.app |

The frontend sends API requests to the Render-hosted FastAPI backend.

### 🤖 Backend — Render

| Setting | Value |
|---|---|
| Build Command | `pip install -r backend/requirements.txt` |
| Start Command | `uvicorn backend.app:app --host 0.0.0.0 --port $PORT` |
| Production API | https://bus-complaint-ai-api.onrender.com |
| Swagger Docs | https://bus-complaint-ai-api.onrender.com/docs |
| Health Check | https://bus-complaint-ai-api.onrender.com/api/health |

> The Render free web-service tier may spin down after inactivity. The next request may therefore take longer while the backend starts.

### 🔄 Deployment Workflow

Both the frontend and backend are connected to the GitHub repository.

```text
Developer
    │
    │  git push
    ▼
GitHub main
    │
    ├─────────────────────┐
    │                     │
    ▼                     ▼
 Vercel                Render
    │                     │
    ▼                     ▼
Frontend Build       Backend Build
    │                     │
    ▼                     ▼
Live Website         FastAPI API
```

A push to the `main` branch can automatically trigger a new deployment for the connected service.

---

## 📈 Dataset and Evaluation

The supplied department-classification dataset contains:

- **48** labeled complaint examples
- **6** departments
- **8** examples per department

Because the dataset is very small, evaluation results should be treated as academic demonstrations rather than evidence of production performance.

The original implementation reported:

- **75%** accuracy on its train/test split
- **66.67%** accuracy for its Naive Bayes baseline

These figures are retained as historical project context. They should **not** be interpreted as a reliable estimate of real-world model performance.

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

## ⚠️ Limitations

TransitIQ is an academic NLP project, **not** a production-grade transit decision system.

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

The system is intended for **demonstration, learning, experimentation and academic evaluation**.

---

## 🔮 Future Improvements

- Expand the labeled complaint dataset
- Add multilingual complaint support (Marathi / Hindi / English language detection)
- Improve sentiment classification
- Train a dedicated urgency classifier
- Compare Logistic Regression with SVM and Naive Bayes
- Experiment with transformer-based models
- Add cross-validation, automated evaluation, confusion matrices and per-class metrics
- Add human-review feedback
- Add model versioning
- Add database-backed complaint history
- Add authentication and role-based access
- Add production monitoring
- Add automated retraining pipelines
- Add explainable model visualizations
- Introduce transformer embeddings for semantic similarity
- Improve duplicate and near-duplicate detection

---

## 🛠️ Technology Stack

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
| Frontend Deployment | Vercel |
| Backend Deployment | Render |
| Version Control | Git + GitHub |

---

## 🔁 Reproducibility

The main model-training workflow is included in the repository.

```bash
# 1. Clone the project
git clone https://github.com/ghawatesiddharth/Bus-Complaint-AI.git
cd Bus-Complaint-AI

# 2. Create and activate the environment
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

# 3. Install dependencies
pip install -r backend/requirements.txt

# 4. Train the model
python -m backend.train

# 5. Run the backend
uvicorn backend.app:app --reload --port 8000
```

Then, in another terminal:

```bash
cd frontend
npm install
npm run dev
```

---

## 🎓 Academic Project Note

TransitIQ was developed as a B-Tech academic project to demonstrate an end-to-end Natural Language Processing workflow.

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

### 📌 Project Goals

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

### ⭐ Project Summary

TransitIQ demonstrates an end-to-end machine learning application that transforms raw passenger complaints into structured, explainable routing information.

```text
Passenger Complaint
        ↓
React Frontend
        ↓
FastAPI API
        ↓
Text Processing
        ↓
Word + Character TF-IDF
        ↓
Logistic Regression
        ↓
Department Prediction
        ↓
Confidence + Sentiment + Urgency
        ↓
Explainable Complaint Analysis
```

---

## 📜 License

No open-source license has been specified for this repository. Unless a license is added, default copyright rules apply to the original project code and content.

---

## 👨‍💻 Project Links

**TransitIQ — Bus Complaint AI** · Built as an academic B-Tech Natural Language Processing project.

| | |
|---|---|
| 💻 **GitHub** | https://github.com/ghawatesiddharth/Bus-Complaint-AI |
| 🌐 **Live Application** | https://bus-complaint-ai.vercel.app |
| 🤖 **Backend API** | https://bus-complaint-ai-api.onrender.com/api/health |
| 📚 **API Documentation** | https://bus-complaint-ai-api.onrender.com/docs |

**Frontend:** Vercel · **Backend:** Render · **Source Code:** GitHub
