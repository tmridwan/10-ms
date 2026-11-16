"""
Run this AFTER running 10ms_viral_classifier.ipynb
It saves the trained model + encoders into viral_model.pkl
so the agent can load it.

Place this file in the same folder as 10ms_videos.csv and run:
  python save_model.py
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# ── Rebuild everything (same as notebook) ────────────────────────────────────
df = pd.read_csv("10ms_videos.csv")

SUBJECTS = {
    "HSC":        ["hsc", "higher secondary"],
    "SSC":        ["ssc", "secondary"],
    "Physics":    ["physics", "পদার্থ"],
    "Chemistry":  ["chemistry", "রসায়ন"],
    "Biology":    ["biology", "জীববিজ্ঞান"],
    "Math":       ["math", "গণিত", "algebra", "calculus"],
    "English":    ["english", "grammar", "vocabulary"],
    "Bangla":     ["bangla", "বাংলা"],
    "ICT":        ["ict", "programming", "python", "computer"],
    "Admission":  ["admission", "ভর্তি"],
    "Motivation": ["motivation", "motivational", "life"],
    "BCS":        ["bcs", "government job"],
}

def tag_subject(title):
    t = str(title).lower()
    for subj, kws in SUBJECTS.items():
        if any(k in t for k in kws):
            return subj
    return "Other"

def dur_bucket(m):
    if m < 5:   return "short"
    if m < 15:  return "medium"
    if m < 30:  return "long"
    return "very_long"

df["subject"]      = df["title"].apply(tag_subject)
df["published"]    = pd.to_datetime(df["published"])
df["upload_month"] = df["published"].dt.month
df["upload_dow"]   = df["published"].dt.dayofweek
df["upload_year"]  = df["published"].dt.year
df["days_since_pub"] = (pd.Timestamp.today() - df["published"]).dt.days
df["like_rate"]    = df["likes"] / df["views"].replace(0, 1)
df["dur_bucket"]   = df["duration_min"].apply(dur_bucket)

threshold = df["views"].quantile(0.90)
df["viral"] = (df["views"] >= threshold).astype(int)

le_subj = LabelEncoder()
le_dur  = LabelEncoder()
df["subject_enc"]    = le_subj.fit_transform(df["subject"])
df["dur_bucket_enc"] = le_dur.fit_transform(df["dur_bucket"])

FEATURES = ["duration_min", "subject_enc", "upload_month", "upload_dow",
            "upload_year", "dur_bucket_enc", "like_rate", "days_since_pub"]

X = df[FEATURES]
y = df["viral"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05,
    num_leaves=31, class_weight="balanced",
    random_state=42, verbose=-1
)
model.fit(X_train, y_train)

# ── Save bundle ───────────────────────────────────────────────────────────────
bundle = {
    "model":     model,
    "le_subj":   le_subj,
    "le_dur":    le_dur,
    "threshold": threshold,
    "features":  FEATURES,
}
joblib.dump(bundle, "viral_model.pkl")
print(f"✅ Model saved to viral_model.pkl")
print(f"   Viral threshold : {threshold:,.0f} views")
print(f"   Subjects known  : {list(le_subj.classes_)}")
