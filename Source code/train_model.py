import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

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

    # --- Required Columns ---
    req_cols = ["degree", "major", "cgpa", "skills", "job_role"]
    for c in req_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column in dataset: {c}")

    # --- Preprocessing ---
    df = df.dropna(subset=["job_role"])
    df["degree"] = df["degree"].fillna("Unknown").astype(str).str.strip()
    df["major"] = df["major"].fillna("Unknown").astype(str).str.strip()
    df["cgpa"] = pd.to_numeric(df["cgpa"], errors="coerce")
    median_cgpa = df["cgpa"].median()
    df["cgpa"] = df["cgpa"].fillna(median_cgpa)
    df = df.dropna(subset=["skills"])
    df["skills"] = df["skills"].astype(str).apply(
        lambda s: [t.strip().lower() for t in s.split(",") if t.strip()]
    )
    df = df[df["skills"].map(len) > 0]

    # --- Feature Encoding ---
    ohe = _make_ohe()
    X_degmaj = ohe.fit_transform(df[["degree", "major"]])

    mlb = MultiLabelBinarizer()
    X_skills = mlb.fit_transform(df["skills"])
    mlb.classes_ = np.array([c.lower() for c in mlb.classes_])

    X_cgpa = df[["cgpa"]].to_numpy().reshape(-1, 1)
    scaler = StandardScaler()
    X_cgpa_scaled = scaler.fit_transform(X_cgpa)

    X = np.hstack([X_degmaj, X_cgpa_scaled, X_skills])
    y = df["job_role"].astype(str).to_numpy()

    # --- Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Apply SMOTE to handle class imbalance ---
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # --- RandomForest with class weights and hyperparameter tuning ---
    rf = RandomForestClassifier(random_state=42, class_weight="balanced")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "bootstrap": [True, False],
    }

    grid = GridSearchCV(
        rf,
        param_grid,
        cv=3,
        scoring="accuracy",
        verbose=1,
        n_jobs=-1
    )
    grid.fit(X_train_res, y_train_res)
    clf = grid.best_estimator_
    print(f"✅ Best Parameters: {grid.best_params_}")

    # --- Evaluate Model ---
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Model Accuracy: {acc*100:.2f}%\n")
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("🔢 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # --- Save model and encoders ---
    joblib.dump(clf, MODEL_PATH)
    joblib.dump({"ohe": ohe, "mlb": mlb, "scaler": scaler}, ENCODERS_PATH)

    print("\n✅ Training complete.")
    print(f"Original dataset rows: {original_rows}")
    print(f"Final dataset rows used for training: {len(df)}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Encoders saved to: {ENCODERS_PATH}")

if __name__ == "__main__":
    train_model()
