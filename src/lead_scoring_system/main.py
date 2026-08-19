import os
import smtplib
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from pymongo import MongoClient

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.mongodb import MongoDBJobStore

# ---------- Environment Variables Load Karo ----------
load_dotenv()

app = FastAPI(title="AI Lead Scoring System")

# ---------- Gemini Client Setup ----------
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ---------- MongoDB Connection ----------
mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client["lead_scoring_db"]

leads_collection = db["leads"]              # Sab leads
hot_leads_collection = db["hot_leads"]      # Sirf Hot leads
sent_emails_collection = db["sent_emails"]  # Bheje gaye emails ka record

# ---------- Persistent Scheduler Setup (MongoDB Job Store) ----------
# NOTE: Server ko bina --reload ke chalao (uvicorn lead_scoring_system.main:app)
# warna --reload do processes banata hai aur scheduler duplicate chal sakta hai.
jobstores = {
    'default': MongoDBJobStore(
        database='lead_scoring_db',
        collection='scheduled_email_jobs',
        client=mongo_client
    )
}
scheduler = BackgroundScheduler(jobstores=jobstores, timezone=timezone.utc)
scheduler.start()

# ---------- Model Aur Encoders Load Karo ----------
ML_DIR = Path(__file__).resolve().parent / "ml"

model = joblib.load(ML_DIR / "lead_score_model.pkl")
industry_encoder = joblib.load(ML_DIR / "industry_encoder.pkl")
job_role_encoder = joblib.load(ML_DIR / "job_role_encoder.pkl")
feature_columns = joblib.load(ML_DIR / "feature_columns.pkl")


# ---------- Request Body Ka Format ----------
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


# ---------- Priority Decide Karne Ka Function ----------
def get_priority(score: float) -> str:
    if score >= 80:
        return "Hot"
    elif score >= 50:
        return "Warm"
    else:
        return "Cold"


# ---------- Gemini Se Reason Generate Karna ----------
def get_reason(lead: LeadInput, score: float, priority: str) -> str:
    prompt = f"""
You are a sales assistant. Based on this lead's data, write ONE short sentence (max 20 words) explaining why this lead got this score and priority.

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
        print("Gemini reason error:", e)
        return f"Lead priority is {priority} based on budget, engagement, and demo interest."


# ---------- Gemini Se Email Content Generate Karna ----------
def generate_email_content(lead_data: dict) -> dict:
    prompt = f"""
You are a friendly B2B sales representative. Write a short, personalized sales
follow-up email for this lead who showed strong buying interest.

Lead details:
- Name: {lead_data['name']}
- Company: {lead_data['company']}
- Industry: {lead_data['industry']}
- Budget: {lead_data['budget']}
- Demo requested: {lead_data['demo_requested']}

Requirements:
- Keep it under 100 words
- Friendly, professional tone
- End with a clear call to action (book a call/demo)

Reply in EXACTLY this format:
SUBJECT: <subject line>
BODY: <email body>
"""
    try:
        response = gemini_client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )
        text = response.text.strip()

        subject = "We'd love to help your business grow"
        body = text

        if "SUBJECT:" in text and "BODY:" in text:
            subject = text.split("SUBJECT:")[1].split("BODY:")[0].strip()
            body = text.split("BODY:")[1].strip()

        return {"subject": subject, "body": body}
    except Exception as e:
        print("Gemini email error:", e)
        return {
            "subject": "Let's connect!",
            "body": f"Hi {lead_data['name']}, thanks for your interest. "
                    f"We'd love to show you how we can help {lead_data['company']}. "
                    f"Let us know a good time to connect!"
        }


# ---------- Email Bhejne Ka Function (SMTP) ----------
def send_email_smtp(to_email: str, subject: str, body: str) -> bool:
    try:
        sender_email = os.getenv("GMAIL_ADDRESS")
        sender_password = os.getenv("GMAIL_APP_PASSWORD")

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = to_email

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_email, msg.as_string())

        return True
    except Exception as e:
        print("SMTP send error:", e)
        return False


# ---------- Ye Function 30 Min Baad Scheduler Se Automatically Chalega ----------
def send_hot_lead_email(lead_data: dict):
    print(f"Sending scheduled email to hot lead: {lead_data['email']}")

    email_content = generate_email_content(lead_data)
    success = send_email_smtp(lead_data["email"], email_content["subject"], email_content["body"])

    sent_emails_collection.insert_one({
        "lead_name": lead_data["name"],
        "lead_email": lead_data["email"],
        "company": lead_data["company"],
        "lead_score": lead_data["lead_score"],
        "priority": lead_data["priority"],
        "email_subject": email_content["subject"],
        "email_body": email_content["body"],
        "sent_at": datetime.now(timezone.utc),
        "status": "sent" if success else "failed",
    })


# ---------- Main Scoring Endpoint ----------
@app.post("/score_lead")
def score_lead(lead: LeadInput):
    try:
        if lead.industry in industry_encoder.classes_:
            industry_encoded = industry_encoder.transform([lead.industry])[0]
        else:
            industry_encoded = 0

        if lead.job_role in job_role_encoder.classes_:
            job_role_encoded = job_role_encoder.transform([lead.job_role])[0]
        else:
            job_role_encoded = 0

        features = np.array([[
            lead.company_size,
            lead.budget,
            industry_encoded,
            job_role_encoded,
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

        # ---------- "leads" Collection Mein Save Karo (Sabka Data) ----------
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

        # ---------- Agar Hot Hai To "hot_leads" Mein Save + Email Schedule Karo ----------
        if priority == "Hot":
            hot_leads_collection.insert_one(lead_record.copy())

            delay_minutes = int(os.getenv("EMAIL_DELAY_MINUTES", 30))
            run_time = datetime.now(timezone.utc) + timedelta(minutes=delay_minutes)

            # MongoDB job store ke liye data plain dict/serializable hona chahiye
            email_job_data = {
                "name": lead.name,
                "email": lead.email,
                "company": lead.company,
                "industry": lead.industry,
                "budget": lead.budget,
                "demo_requested": lead.demo_requested,
                "lead_score": predicted_score,
                "priority": priority,
            }

            scheduler.add_job(
                send_hot_lead_email,
                'date',
                run_date=run_time,
                args=[email_job_data],
                id=f"email_{lead.email}_{datetime.now(timezone.utc).timestamp()}",
                replace_existing=False,
            )

        return {
            "status": True,
            "lead_score": predicted_score,
            "priority": priority,
            "reason": reason,
        }

    except Exception as e:
        return {"status": False, "error": str(e)}


# ---------- Saare Leads Dekhne Ka Endpoint ----------
@app.get("/leads")
def get_all_leads():
    leads = list(leads_collection.find({}, {"_id": 0}).sort("lead_score", -1))
    return {"status": True, "count": len(leads), "leads": leads}


# ---------- Sirf Hot Leads Dekhne Ka Endpoint ----------
@app.get("/leads/hot")
def get_hot_leads():
    leads = list(hot_leads_collection.find({}, {"_id": 0}).sort("lead_score", -1))
    return {"status": True, "count": len(leads), "leads": leads}


# ---------- Bheje Gaye Emails Dekhne Ka Endpoint ----------
@app.get("/emails/sent")
def get_sent_emails():
    emails = list(sent_emails_collection.find({}, {"_id": 0}).sort("sent_at", -1))
    return {"status": True, "count": len(emails), "emails": emails}


# ---------- Health Check ----------
@app.get("/")
def read_root():
    return {"message": "Lead Scoring System API is running!"}