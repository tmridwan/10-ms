"""
10 Minute School — Viral Video Prediction Agent
================================================
LangChain Agent + LightGBM + Gradio — hosted on Hugging Face Spaces

Environment variable required (set in HF Space secrets):
  GEMINI_API_KEY = your_gemini_api_key
"""

import os, joblib, re
import pandas as pd
import numpy as np
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
import warnings
warnings.filterwarnings("ignore")
import gradio as gr

# ── Load model bundle ─────────────────────────────────────────────────────────
bundle    = joblib.load("viral_model.pkl")
model     = bundle["model"]
le_subj   = bundle["le_subj"]
le_dur    = bundle["le_dur"]
threshold = bundle["threshold"]

# ── Helpers ───────────────────────────────────────────────────────────────────
SUBJECTS = {
    "HSC":        ["hsc", "higher secondary"],
    "SSC":        ["ssc", "secondary"],
    "Physics":    ["physics"],
    "Chemistry":  ["chemistry"],
    "Biology":    ["biology"],
    "Math":       ["math", "algebra", "calculus"],
    "English":    ["english", "grammar", "vocabulary"],
    "Bangla":     ["bangla"],
    "ICT":        ["ict", "programming", "python", "computer"],
    "Admission":  ["admission"],
    "Motivation": ["motivation", "life"],
    "BCS":        ["bcs", "government job"],
}

def tag_subject(title):
    t = title.lower()
    for subj, kws in SUBJECTS.items():
        if any(k in t for k in kws):
            return subj
    return "Other"

def dur_bucket(m):
    if m < 5:   return "short"
    if m < 15:  return "medium"
    if m < 30:  return "long"
    return "very_long"

def predict_viral_score(title, duration_min, upload_month, upload_dow=1, upload_year=2025):
    subject  = tag_subject(title)
    subj_enc = le_subj.transform([subject])[0] if subject in le_subj.classes_ else 0
    dur_b    = dur_bucket(duration_min)
    dur_enc  = le_dur.transform([dur_b])[0]
    days     = (pd.Timestamp.today() - pd.Timestamp(f"{upload_year}-{upload_month:02d}-01")).days

    sample = pd.DataFrame([{
        "duration_min":   duration_min,
        "subject_enc":    subj_enc,
        "upload_month":   upload_month,
        "upload_dow":     upload_dow,
        "upload_year":    upload_year,
        "dur_bucket_enc": dur_enc,
        "like_rate":      0.04,
        "days_since_pub": days,
    }])

    prob  = model.predict_proba(sample)[0][1]
    label = "VIRAL 🔥" if prob >= 0.5 else "Normal 📉"
    return {
        "subject": subject,
        "duration_bucket": dur_b,
        "viral_probability": round(prob * 100, 1),
        "prediction": label,
        "threshold_views": int(threshold),
    }

# ── LangChain Tool ────────────────────────────────────────────────────────────
def viral_tool_fn(query):
    dur_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:min|minute)", query, re.I)
    duration  = float(dur_match.group(1)) if dur_match else 10.0

    month_map = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
                 "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
    month = 6
    for name, num in month_map.items():
        if name in query.lower():
            month = num
            break
    num_month = re.search(r"\bmonth\s+(\d{1,2})\b", query, re.I)
    if num_month:
        month = int(num_month.group(1))

    dow_map = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,
               "friday":4,"saturday":5,"sunday":6}
    dow = 1
    for day, num in dow_map.items():
        if day in query.lower():
            dow = num
            break

    r = predict_viral_score(query, duration, month, dow)
    return (
        f"Prediction: {r['prediction']}\n"
        f"Viral probability: {r['viral_probability']}%\n"
        f"Subject detected: {r['subject']}\n"
        f"Duration bucket: {r['duration_bucket']}\n"
        f"Viral threshold: >{r['threshold_views']:,} views (top 10% of 10MS videos)\n"
    )

viral_tool = Tool(
    name="ViralVideoPrediction",
    func=viral_tool_fn,
    description=(
        "Predicts whether a 10 Minute School YouTube video will go viral (top 10% views). "
        "Input describes the video: topic, duration in minutes, upload month. "
        "Use this for any question about video virality, views prediction, or upload strategy."
    )
)

# ── LangChain Agent ───────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.environ["GEMINI_API_KEY"],
    temperature=0.3,
)

agent = initialize_agent(
    tools=[viral_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=4,
)

SYSTEM_CONTEXT = (
    "You are an AI analyst for 10 Minute School, Bangladesh's largest online education platform. "
    "You help content creators predict whether their YouTube videos will go viral. "
    "You have a machine learning model trained on 2,845 real 10 Minute School videos. "
    "Always use the ViralVideoPrediction tool when asked about video performance. "
    "Give clear, actionable advice."
)

# ── Gradio UI ─────────────────────────────────────────────────────────────────
def chat(user_message, history):
    try:
        response = agent.run(f"{SYSTEM_CONTEXT}\n\nUser: {user_message}")
    except Exception as e:
        response = f"Sorry, I hit an error: {str(e)}\n\nTry including the duration (e.g. '15 minutes') and a month."
    history.append((user_message, response))
    return history, ""

with gr.Blocks(title="10MS Viral Predictor") as demo:
    gr.Markdown("""
# 🎯 10 Minute School — Viral Video Prediction Agent
**ML model trained on 2,845 real 10MS YouTube videos · LightGBM + LangChain + Gemini**

Try asking:
- *Will a 15-minute ICT video uploaded in June go viral?*
- *What subject and length gives the best chance of going viral?*
- *Should I upload a 45-minute HSC Physics lecture on Friday in January?*
    """)

    chatbot = gr.Chatbot(height=450)
    msg     = gr.Textbox(placeholder="Ask about your video idea...", label="Your question", lines=1)
    clear   = gr.Button("🗑 Clear chat")

    msg.submit(chat, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: ([], ""), outputs=[chatbot, msg])

demo.launch(theme=gr.themes.Soft(), ssr_mode=False)
