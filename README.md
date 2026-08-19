# AI Lead Scoring System

An AI/ML-powered backend application that analyzes B2B sales leads and assigns a numerical score (0–100) along with a priority label (**Hot / Warm / Cold**), helping sales teams focus on the leads most likely to convert.

---

## 📌 Overview

Sales teams often receive dozens of leads daily but have no reliable way to know which ones deserve immediate attention. This project solves that problem by combining a trained **Machine Learning model** with the **Gemini LLM API** to:

1. Predict a lead's conversion likelihood as a numeric score.
2. Classify the lead into a priority tier (Hot / Warm / Cold).
3. Generate a short, human-readable explanation of *why* the lead received that score.

---

## 🚀 Features

- REST API built with **FastAPI** for real-time lead scoring
- **Gradient Boosting Regressor** (scikit-learn) trained on lead data to predict scores
- Automatic **priority classification** (Hot / Warm / Cold) based on score thresholds
- **AI-generated explanations** using Google's Gemini API for natural, dynamic reasoning
- Fallback rule-based reasoning if the Gemini API is unavailable
- Clean, testable API — verified with Postman

---

## 🧠 How It Works

1. A lead's information (company size, budget, industry, engagement, etc.) is sent to the `/score_lead` endpoint.
2. The backend validates the incoming data using Pydantic models.
3. Categorical fields (`industry`, `job_role`) are encoded using saved `LabelEncoder`s.
4. The trained ML model predicts a `lead_score` (0–100).
5. The score is mapped to a priority:
   - **80–100 → Hot**
   - **50–79 → Warm**
   - **0–49 → Cold**
6. The Gemini API generates a short, natural-language reason for the score.
7. The API returns `lead_score`, `priority`, and `reason` as a JSON response.

---

## 🗂️ Tech Stack

| Layer | Technology |
|---|---|
| Backend Framework | Python + FastAPI |
| ML Library | scikit-learn (Gradient Boosting Regressor) |
| AI Reasoning | Google Gemini API |
| Package Manager | uv |
| Data Handling | pandas, numpy |
| Model Persistence | joblib |
| Testing | Postman |

---

## 📁 Project Structure

```
lead-scoring-system/
├── src/
│   └── lead_scoring_system/
│       ├── main.py                    # FastAPI application & API endpoints
│       ├── __init__.py
│       ├── data/
│       │   └── leads_dataset.csv      # Training dataset
│       ├── ml/
│       │   ├── lead_score_model.pkl       # Trained ML model
│       │   ├── industry_encoder.pkl       # Saved LabelEncoder for industry
│       │   ├── job_role_encoder.pkl       # Saved LabelEncoder for job role
│       │   ├── feature_columns.pkl        # Feature order used during training
│       │   └── model_info.txt             # Best model name & performance metrics
│       └── Notebook/
│           └── eda.ipynb              # Data exploration & model training notebook
├── .env                                # Environment variables (API keys) — not committed
├── pyproject.toml                      # Project dependencies (uv)
├── uv.lock
└── README.md
```

---

## 📊 Dataset

The dataset contains 1,000 synthetic B2B leads with the following fields:

| Column | Description |
|---|---|
| `company_size` | Number of employees at the lead's company |
| `budget` | Lead's stated budget |
| `industry` | Industry the lead's company operates in |
| `job_role` | Lead's job role / decision-making authority |
| `engagement_score` | Website/product engagement (0–10) |
| `email_interaction_score` | Email engagement score |
| `previous_interactions` | Number of prior interactions with the company |
| `demo_requested` | Whether the lead requested a product demo |
| `previous_purchase` | Whether the lead has purchased before |
| `lead_score` | **Target variable** — the score the model learns to predict |
| `priority` | Hot / Warm / Cold label (derived from `lead_score`) |

`name`, `company`, and `priority` are dropped before training since they are identifiers or a derived label, not predictive features.

---

## 🤖 Model Training

Three regression models were trained and compared using an 80/20 train-test split and 5-fold cross-validation:

| Model | MAE | RMSE | R² Score | CV R² (mean) |
|---|---|---|---|---|
| **Gradient Boosting** ✅ | 3.72 | 4.63 | **0.925** | 0.914 |
| Random Forest | 4.82 | 5.96 | 0.876 | 0.870 |
| Linear Regression | 6.14 | 7.71 | 0.792 | 0.733 |

**Gradient Boosting Regressor** was selected as the final model based on the highest R² score and lowest error metrics, with cross-validation confirming the result is not overfit.

---

## ⚙️ Setup Instructions

### 1. Clone the project and install dependencies

```bash
uv venv
.venv\Scripts\activate      # Windows
uv add fastapi uvicorn scikit-learn pandas numpy joblib python-dotenv google-genai
```

### 2. Add your Gemini API key

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).

### 3. Train the model (if not already trained)

Run all cells in `src/lead_scoring_system/Notebook/eda.ipynb` to preprocess the data, train the models, and save the best one to the `ml/` folder.

### 4. Run the API server

```bash
uv run uvicorn lead_scoring_system.main:app --reload
```

The server will start at: `http://127.0.0.1:8000`

Interactive API docs are available at: `http://127.0.0.1:8000/docs`

---

## 📡 API Reference

### `POST /score_lead`

Scores a single lead and returns its priority and reasoning.

**Request Body:**

```json
{
  "name": "Priya",
  "company": "TechNova Solutions",
  "company_size": 250,
  "budget": 400000,
  "industry": "Software",
  "job_role": "Manager",
  "engagement_score": 8,
  "email_interaction_score": 7,
  "previous_interactions": 5,
  "demo_requested": true,
  "previous_purchase": false
}
```

**Response:**

```json
{
  "status": true,
  "lead_score": 87.3,
  "priority": "Hot",
  "reason": "This lead is Hot due to a large company size, high budget, and demo request, indicating strong purchase intent."
}
```

### `GET /`

Health check endpoint — confirms the API is running.

```json
{ "message": "Lead Scoring System API is running!" }
```

---

## 🧪 Testing

The API was tested using **Postman** with multiple lead profiles (high-budget enterprise leads, low-engagement small leads, etc.) to verify that scores, priorities, and reasons are generated correctly across different scenarios.

---

## 🔮 Future Enhancements

- Store leads and score history in a database (MongoDB/PostgreSQL)
- Build a sales dashboard to view and sort leads by score
- Add authentication for API access
- Automatic email notifications for Hot leads
- CRM integration
- Periodic model retraining with new lead data

---

## 👤 Author

Built as part of an internship project to demonstrate an end-to-end AI/ML backend system — from data preprocessing and model training to API deployment and AI-powered reasoning.
