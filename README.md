<div align="center">

# 🌟 Edu2Job - AI-Powered Career Prediction Platform  

🚀 *Empowering Students to Find the Right Career Path through AI Insights*  

</div>

---

## 🧠 Overview  

**Edu2Job** is an intelligent career prediction web app that suggests ideal **job roles** based on your **education, skills, and academic background**.  
It provides an interactive dashboard, real-time prediction history, and personalized insights — all powered by **Machine Learning**.

---

## ✨ Features  

✅ Secure **User Authentication** (JWT-based)  
✅ Personalized **Profile Management** with image upload  
✅ **AI-powered Job Prediction** using Decision Tree / Random Forest models  
✅ Comprehensive **Prediction History** with timestamps & confidence levels  
✅ Sleek **Interactive Dashboard** featuring:  
   - 📊 Total Predictions  
   - 🧭 Confidence Level Analysis (High / Medium / Low)  
   - 🧠 Top Predicted Roles  
   - 🕒 Recent Activity Feed  
✅ Responsive & modern **Tailwind CSS UI**  
✅ Skill input with **autocomplete + tag support**  
✅ **College selection** from verified Indian colleges list  

---

## 🛠️ Tech Stack  

| Layer | Technologies |
|-------|---------------|
| **Frontend** | HTML, Tailwind CSS, JavaScript |
| **Backend** | Python, Flask, SQLAlchemy |
| **Database** | SQLite |
| **ML Models** | Scikit-learn (Decision Tree / Random Forest) |
| **Libraries** | pandas, pickle, Flask-CORS |
| **Version Control** | Git & GitHub |

---

## ⚙️ Getting Started  

### 📋 Prerequisites  
Ensure you have **Python 3.10+** and `pip` installed.  

### 🧩 Installation  

```bash
# Clone the repository
git clone https://github.com/SreehariniJ/Edu2Job.git
cd Edu2Job/Source\ code

# Create and activate virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Then open your browser and visit:  
👉 **http://127.0.0.1:5000/**

---

## 🎯 How to Use  

1. Register or log in using your email credentials.  
2. Complete your profile (degree, major, skills, college, etc.).  
3. Go to **“New Prediction”** to generate a personalized job prediction.  
4. Explore **“Prediction History”** for detailed analytics & insights.  

---

## 🗂️ Project Structure  

```
Edu2Job/
├── Source code/
│   ├── app.py
│   ├── train_model.py
│   ├── edu2job.db
│   ├── encoders.pkl
│   ├── model.pkl
│   ├── requirements.txt
│   ├── static/
│   │   ├── default_user.png
│   │   └── uploads/
│   └── templates/
│       ├── index.html
│       ├── dashboard.html
│       ├── profile.html
│       ├── new_prediction.html
│       └── history.html
└── README.md
```

---

## 🤝 Contributing  

We welcome all contributions! 🎉  
Fork the repository, create a new branch, and submit a **pull request** with your improvements.

---

## 📜 License  

Licensed under the **MIT License**.  
You are free to use, modify, and distribute this project responsibly.

---

## 👩‍💻 Author  

**Sreeharini J**  
📧 Email: [sreeharinij@gmail.com](mailto:sreeharinij@gmail.com)  
💻 GitHub: [github.com/SreehariniJ](https://github.com/SreehariniJ)  

---

<div align="center">

⭐ *If you like this project, give it a star on GitHub!* ⭐

</div>
