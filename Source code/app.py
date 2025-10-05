# app.py
import os
import datetime
import joblib
import bcrypt
import jwt
import numpy as np
import pandas as pd
from functools import wraps
from flask import Flask, request, jsonify, render_template, current_app
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import re

# ---------------- Config ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "edu2job.db")
DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")

SECRET_KEY = "change_this_secret"  # 🔐 replace in production
JWT_ALGO = "HS256"
JWT_EXP_HOURS = 4

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Flask app setup
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# ---------------- Database Models ----------------
class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(200), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    degree = db.Column(db.String(120))
    major = db.Column(db.String(120))
    cgpa = db.Column(db.Float)
    graduation_year = db.Column(db.Integer)
    skills = db.Column(db.String(1000))
    phone = db.Column(db.String(50))
    country_code = db.Column(db.String(10))
    college = db.Column(db.String(200))
    profile_photo = db.Column(db.String(300))  # store filename


class PredictionHistory(db.Model):
    __tablename__ = "prediction_history"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    degree = db.Column(db.String(120))
    major = db.Column(db.String(120))
    cgpa = db.Column(db.Float)
    skills = db.Column(db.String(1000))
    predicted_role = db.Column(db.String(300))
    confidence = db.Column(db.Float)
    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.datetime.now(datetime.timezone.utc)
    )


# Ensure DB exists
with app.app_context():
    db.create_all()


# ---------------- JWT Decorator ----------------
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get("Authorization")
        if not auth:
            return jsonify({"error": "Authorization header missing"}), 401

        parts = auth.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return jsonify({"error": "Invalid Authorization header"}), 401

        token = parts[1]
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGO])
            user_id = payload.get("user_id")
            if not user_id:
                return jsonify({"error": "Invalid token payload"}), 401

            user = User.query.get(user_id)
            if not user:
                return jsonify({"error": "User not found"}), 401
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except Exception:
            return jsonify({"error": "Invalid token"}), 401

        return f(user, *args, **kwargs)

    return decorated


# ---------------- Model Utilities ----------------
def ensure_model():
    """Train model if missing, return True if model exists."""
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH):
        return True
    try:
        from train_model import train_model
        train_model()
    except Exception as e:
        current_app.logger.error(f"Training failed: {e}")
    return os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH)


# Load ML model + encoders
MODEL, OHE, MLB = None, None, None
if ensure_model():
    try:
        MODEL = joblib.load(MODEL_PATH)
        enc = joblib.load(ENCODERS_PATH)
        OHE, MLB = enc.get("ohe"), enc.get("mlb")
        app.logger.info("✅ ML model and encoders loaded.")
    except Exception as e:
        app.logger.error(f"❌ Failed to load model/encoders: {e}")


def get_dataset_options():
    """Return degrees, majors, and skills from dataset."""
    try:
        if not os.path.exists(DATASET_PATH):
            return {"degrees": [], "majors": [], "skills": []}

        df = pd.read_csv(DATASET_PATH, on_bad_lines='skip', encoding='utf-8')

        # Filter out "Unknown" and variations
        def filter_unknown_values(series):
            if series is None or series.empty:
                return []
            # Convert to string, strip whitespace, and filter out unknown variations
            cleaned = series.dropna().astype(str).str.strip()
            # Filter out various forms of "unknown"
            filtered = cleaned[
                ~cleaned.str.lower().isin(['unknown', 'unknown ', ' unknown', 'n/a', 'na', 'none', 'null'])
            ]
            return sorted(filtered.unique().tolist())

        degrees = filter_unknown_values(df["degree"] if "degree" in df.columns else None)
        majors = filter_unknown_values(df["major"] if "major" in df.columns else None)

        skills_set = set()
        if "skills" in df.columns:
            for s in df["skills"].dropna().astype(str):
                # Clean and split skills
                clean_skills = re.sub(r'[\[\]\'\"]', '', s)  # Remove brackets and quotes
                for token in clean_skills.split(','):
                    skill = token.strip().lower()
                    if (skill and len(skill) > 1 and 
                        skill not in ['unknown', 'n/a', 'na', 'none', 'null']):
                        skills_set.add(skill)

        return {
            "degrees": degrees,
            "majors": majors, 
            "skills": sorted(skills_set)
        }
    
    except Exception as e:
        print(f"❌ Error in get_dataset_options: {e}")
        return {"degrees": [], "majors": [], "skills": []}
# ---------------- Routes - Pages ----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")


@app.route("/profile")
def profile_page():
    return render_template("profile.html")


@app.route("/new_prediction")
def new_prediction_page():
    return render_template("new_prediction.html")


@app.route("/history")
def history_page():
    return render_template("history.html")


@app.route("/insights")
def insights_page():
    return render_template("insights.html")


# ---------------- Auth APIs ----------------
@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json(force=True) or {}
    fullname = (data.get("fullname") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not fullname or not email or not password:
        return jsonify({"error": "fullname, email and password required"}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered"}), 400

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    user = User(fullname=fullname, email=email, password_hash=hashed)
    db.session.add(user)
    db.session.commit()

    return jsonify({"message": "Registration successful"}), 200


@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(force=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400

    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({"error": "Invalid credentials"}), 401

    stored = user.password_hash.encode("utf-8") if isinstance(user.password_hash, str) else user.password_hash
    if not bcrypt.checkpw(password.encode("utf-8"), stored):
        return jsonify({"error": "Invalid credentials"}), 401

    token = jwt.encode({
        "user_id": user.id,
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=JWT_EXP_HOURS)
    }, SECRET_KEY, algorithm=JWT_ALGO)

    return jsonify({"message": "Login successful", "token": token}), 200


@app.route("/api/options", methods=["GET"])
def api_options():
    return jsonify(get_dataset_options()), 200


# ---------------- Profile APIs ----------------
@app.route("/api/profile", methods=["GET"])
@token_required
def api_get_profile(current_user):
    profile_photo_url = None
    if current_user.profile_photo:
        profile_photo_url = f"/static/uploads/{current_user.profile_photo}"

    skills_list = (current_user.skills.split(",") if current_user.skills else [])
    return jsonify({
        "fullname": current_user.fullname,
        "email": current_user.email,
        "degree": current_user.degree,
        "major": current_user.major,
        "cgpa": current_user.cgpa,
        "graduation_year": current_user.graduation_year,
        "skills": [s for s in skills_list if s],
        "phone": current_user.phone,
        "country_code": current_user.country_code,
        "college": current_user.college,
        "profile_photo_url": profile_photo_url
    }), 200


@app.route("/api/profile", methods=["POST"])
@token_required
def api_save_profile(current_user):
    try:
        # For FormData (profile photo + other fields)
        data = request.form

        current_user.degree = data.get("degree")
        current_user.major = data.get("major")
        current_user.cgpa = float(data.get("cgpa")) if data.get("cgpa") not in (None, "") else None
        current_user.graduation_year = int(data.get("graduation_year")) if data.get("graduation_year") not in (None, "") else None
        current_user.phone = data.get("phone")
        current_user.country_code = data.get("country_code")
        current_user.college = data.get("college")

        # Skills as tags
        skills_list = data.get("skills", "")
        if isinstance(skills_list, str):
            skills_list = [s.strip() for s in skills_list.split(",") if s.strip()]
        current_user.skills = ", ".join(skills_list) if skills_list else None

        # Handle profile photo upload
        if "profile_photo" in request.files:
            photo = request.files["profile_photo"]
            if photo.filename:
                # Delete old photo if exists
                if current_user.profile_photo:
                    old_path = os.path.join(UPLOAD_FOLDER, current_user.profile_photo)
                    if os.path.exists(old_path):
                        os.remove(old_path)
                filename = f"user_{current_user.id}_{secure_filename(photo.filename)}"
                photo.save(os.path.join(UPLOAD_FOLDER, filename))
                current_user.profile_photo = filename

        # Handle photo removal
        remove_photo_flag = data.get("remove_photo", "false").lower() == "true"
        if remove_photo_flag and current_user.profile_photo:
            old_path = os.path.join(UPLOAD_FOLDER, current_user.profile_photo)
            if os.path.exists(old_path):
                os.remove(old_path)
            current_user.profile_photo = None

        db.session.commit()
        return jsonify({"message": "Profile saved"}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "Save failed: " + str(e)}), 500

# ---------------- Prediction APIs ----------------
@app.route("/api/predict", methods=["POST"])
@token_required
def api_predict(current_user):
    try:
        data = request.get_json(force=True) or {}
        degree = (data.get("degree") or "").strip()
        major = (data.get("major") or "").strip()
        cgpa = float(data.get("cgpa") or 0)
        skills = data.get("skills") or []

        if not degree or not major or not skills:
            return jsonify({"error": "Please fill in all fields."}), 400

        if MODEL is None or OHE is None or MLB is None:
            return jsonify({"error": "Model not loaded"}), 500

        # Prepare input for prediction
        degree_major = [[degree, major]]
        X_degmaj = OHE.transform(degree_major)

        # Handle skills (lowercase)
        skills = [s.lower() for s in skills]
        X_skills = MLB.transform([skills])

        X_cgpa = np.array([[cgpa]])
        X_input = np.hstack([X_degmaj, X_cgpa, X_skills])

        # Predict probabilities
        probs = MODEL.predict_proba(X_input)[0]
        classes = MODEL.classes_
        top_indices = np.argsort(probs)[::-1][:5]

        top_roles = [{"role": classes[i], "confidence": float(probs[i])} for i in top_indices]
        predicted_role = top_roles[0]["role"]
        confidence = top_roles[0]["confidence"]

        # Save to prediction history
        history = PredictionHistory(
            user_id=current_user.id,
            degree=degree,
            major=major,
            cgpa=cgpa,
            skills=", ".join(skills),
            predicted_role=predicted_role,
            confidence=confidence
        )
        db.session.add(history)
        db.session.commit()

        return jsonify({
            "predicted_role": predicted_role,
            "confidence": confidence,
            "top_suggestions": top_roles
        })

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ---------------- History APIs ----------------
@app.route("/api/history", methods=["GET"])
@token_required
def api_history(current_user):
    try:
        items = PredictionHistory.query.filter_by(user_id=current_user.id)\
            .order_by(PredictionHistory.created_at.desc()).all()

        out = [{
            "id": it.id,
            "degree": it.degree,
            "major": it.major,
            "cgpa": it.cgpa,
            "skills": it.skills,
            "predicted_role": it.predicted_role,
            "confidence": float(it.confidence) if it.confidence else None,
            "created_at": it.created_at.isoformat()
        } for it in items]

        current_app.logger.info(f"User {current_user.id} history count: {len(out)}")
        return jsonify(out), 200
    except Exception as e:
        current_app.logger.error(f"History load failed: {e}")
        return jsonify({"error": f"Failed to load history: {e}"}), 500


# ---------------- Run ----------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
