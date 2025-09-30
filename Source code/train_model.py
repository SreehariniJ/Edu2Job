# train_model.py
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")

def _make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def train_model():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    original_rows = len(df)

    # Ensure required columns exist
    req_cols = ["degree", "major", "cgpa", "skills", "job_role"]
    for c in req_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column in dataset: {c}")

    # Drop rows with missing target (job_role)
    before = len(df)
    df = df.dropna(subset=["job_role"]).copy()
    print(f"🗑️ Dropped {before - len(df)} rows with missing job_role")

    # Degree & Major: replace NaN with "Unknown"
    df["degree"] = df["degree"].fillna("Unknown").astype(str).str.strip()
    df["major"] = df["major"].fillna("Unknown").astype(str).str.strip()

    # CGPA: numeric, fill NaN with median
    df["cgpa"] = pd.to_numeric(df["cgpa"], errors="coerce")
    median_cgpa = df["cgpa"].median()
    before = df["cgpa"].isna().sum()
    df["cgpa"] = df["cgpa"].fillna(median_cgpa)
    print(f"📊 Filled {before} missing CGPA values with median = {median_cgpa:.2f}")

    # Skills: drop if missing or empty
    before = len(df)
    df = df.dropna(subset=["skills"])
    dropped_missing_skills = before - len(df)

    df["skills"] = df["skills"].astype(str).apply(
        lambda s: [t.strip().lower() for t in s.split(",") if t.strip()]
    )
    before = len(df)
    df = df[df["skills"].map(len) > 0]  # drop rows where skills list is empty
    dropped_empty_skills = before - len(df)

    print(f"🗑️ Dropped {dropped_missing_skills} rows with missing skills")
    print(f"🗑️ Dropped {dropped_empty_skills} rows with empty skill list")

    # One-hot encode degree + major
    ohe = _make_ohe()
    X_degmaj = ohe.fit_transform(df[["degree", "major"]])

    # MultiLabelBinarizer for skills
    mlb = MultiLabelBinarizer()
    X_skills = mlb.fit_transform(df["skills"])
    mlb.classes_ = np.array([c.lower() for c in mlb.classes_])

    # Numeric feature: cgpa
    X_cgpa = df[["cgpa"]].to_numpy().reshape(-1, 1)

    # Final X
    X = np.hstack([X_degmaj, X_cgpa, X_skills])
    y = df["job_role"].astype(str).to_numpy()

    # Train RandomForest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    # Save model + encoders
    joblib.dump(clf, MODEL_PATH)
    joblib.dump({"ohe": ohe, "mlb": mlb}, ENCODERS_PATH)

    print("\n✅ Training complete.")
    print(f"Original dataset rows: {original_rows}")
    print(f"Final dataset rows used for training: {len(df)}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Encoders saved to: {ENCODERS_PATH}")

if __name__ == "__main__":
    train_model()
