import os
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timezone
import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from pymongo import MongoClient


load_dotenv()

app = FastAPI(title="AI Lead Scoring System")

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

mongo_client = MongoClient(os.getenv("MONGO_URI"))

db = mongo_client["lead_scoring_db"]

leads_collection = db["leads"]
hot_leads_collection = db["hot_leads"]

class LeadInput(BaseModel):
    name: str
    email: str
    company: str
    company_size: int
    budget: float
    industry: str
    job_role: str
    engagement_score: int
    email_interaction_score: int
    previous_interactions: int
    demo_requested: bool
    previous_purchase: bool

ML_DIR = Path(__file__).resolve().parent / "ml"

model = joblib.load(ML_DIR / "lead_score_model.pkl")             
industry_encoder = joblib.load(ML_DIR / "industry_encoder.pkl")   
job_role_encoder = joblib.load(ML_DIR / "job_role_encoder.pkl")  
feature_columns = joblib.load(ML_DIR / "feature_columns.pkl")    

def get_priority(score: float) -> str:
    if score >= 80:
        return "Hot"
    elif score >= 50:
        return "Warm"
    else:
        return "Cold"

def get_reason(lead: LeadInput, score: float, priority: str) -> str:
    prompt = f"""
You are a sales assistant. Based on this lead's data, write ONE short
sentence (max 20 words) explaining why this lead got this score and
priority. Be specific and factual.

Lead details:
- Company size: {lead.company_size} employees
- Budget: {lead.budget}
- Industry: {lead.industry}
- Job role: {lead.job_role}
- Demo requested: {lead.demo_requested}
- Previous purchase: {lead.previous_purchase}
- Engagement score: {lead.engagement_score}/10
- Lead Score: {score}/100
- Priority: {priority}

Reply with ONLY the reason sentence, nothing else.
"""
    try:
        response = gemini_client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print("Gemini error:", e)
        # Fallback: agar Gemini kaam na kare, to bhi user ko
        # khaali reason na jaaye.
        return f"Lead priority is {priority} based on budget, engagement, and demo interest."


@app.post("/score_lead")
def score_lead(lead: LeadInput):
    try:
       
        if lead.industry in industry_encoder.classes_:
            industry_val = industry_encoder.transform([lead.industry])[0]
        else:
            industry_val = 0

        if lead.job_role in job_role_encoder.classes_:
            job_role_val = job_role_encoder.transform([lead.job_role])[0]
        else:
            job_role_val = 0

        features = np.array([[
            lead.company_size,
            lead.budget,
            industry_val,
            job_role_val,
            lead.engagement_score,
            lead.email_interaction_score,
            lead.previous_interactions,
            int(lead.demo_requested),      
            int(lead.previous_purchase),   
        ]])

        predicted_score = model.predict(features)[0]
        predicted_score = round(float(np.clip(predicted_score, 0, 100)), 1)

        priority = get_priority(predicted_score)
        reason = get_reason(lead, predicted_score, priority)

        lead_record = {
            "name": lead.name,
            "email": lead.email,
            "company": lead.company,
            "company_size": lead.company_size,
            "budget": lead.budget,
            "industry": lead.industry,
            "job_role": lead.job_role,
            "engagement_score": lead.engagement_score,
            "email_interaction_score": lead.email_interaction_score,
            "previous_interactions": lead.previous_interactions,
            "demo_requested": lead.demo_requested,
            "previous_purchase": lead.previous_purchase,
            "lead_score": predicted_score,
            "priority": priority,
            "reason": reason,
            "created_at": datetime.now(timezone.utc),
        }

        leads_collection.insert_one(lead_record.copy())

       
        if priority == "Hot":
            hot_leads_collection.insert_one(lead_record.copy())

       
        return {
            "status": True,
            "lead_score": predicted_score,
            "priority": priority,
            "reason": reason,
        }

    except Exception as e:
        
        return {"status": False, "error": str(e)}


@app.get("/leads")
def get_all_leads():
    leads = list(leads_collection.find({}, {"_id": 0}).sort("lead_score", -1))
    return {"status": True, "count": len(leads), "leads": leads}

@app.get("/leads/hot")
def get_hot_leads():
    leads = list(hot_leads_collection.find({}, {"_id": 0}).sort("lead_score", -1))
    return {"status": True, "count": len(leads), "leads": leads}

@app.get("/")
def read_root():
    return {"message": "Lead Scoring System API is running!"}