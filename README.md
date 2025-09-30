# Edu2Job - AI-Powered Career Prediction Platform

Edu2Job is a web application that predicts suitable job roles based on a user's educational background, skills, and other profile details. It provides users with a personalized dashboard, prediction history, and insights using AI models.

---

## Features

- User registration and authentication with JWT.
- Profile management with photo upload, skills, college, and personal details.
- AI-powered job role prediction based on profile data.
- Prediction history with timestamps and confidence levels.
- Interactive dashboard with stats cards for:
  - Total Predictions
  - Predictions by Confidence Level (High / Medium / Low)
  - Most Predicted Roles
  - Recent Activity Feed
- Responsive and visually appealing UI using Tailwind CSS.
- Skills input with tag support and autocomplete.
- College selection from a predefined list of Indian colleges.

---

## Technologies Used

- **Backend:** Python, Flask, SQLAlchemy
- **Database:** SQLite
- **Frontend:** HTML, Tailwind CSS, JavaScript
- **Machine Learning:** Scikit-learn (Decision Tree / Random Forest models)
- **Other Libraries:** pandas, pickle, Flask-CORS
- **Version Control:** Git & GitHub

---

## Getting Started

### Prerequisites

Make sure you have Python 3.10+ installed. You will also need `pip` for installing dependencies.

### Installation

1. Clone the repository:

```bash
git clone https://github.com/SreehariniJ/Edu2Job.git
cd Edu2Job/Source\ code
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Flask app:

```bash
python app.py
```

5. Open your browser and navigate to:

```
http://127.0.0.1:5000/
```

---

## Usage

1. Register or login using your email.
2. Complete your profile with degree, major, skills, and college.
3. Navigate to “New Prediction” to get a job role prediction based on your profile.
4. Check “Prediction History” to view past predictions and insights.

---

## Project Structure

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

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

---

## License

This project is licensed under the MIT License.

---

## Contact

**Sreeharini J**  
Email: sreeharinij@gmail.com  
GitHub: [https://github.com/SreehariniJ]
