# ResumeAI Pro — ATS Resume Screening & Optimization Tool

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![NLP](https://img.shields.io/badge/NLP-Sentence%20Transformers-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

ResumeAI Pro is an ATS-inspired resume screening web app that compares a resume against a job description and generates actionable improvement insights.  
It combines semantic similarity matching with a skill-database checklist to help tailor resumes for ATS and recruiter review.

---

## Live Demo
- **App:** https://resume-parser-ai-aryan-dahiya.streamlit.app/
- **GitHub:** https://github.com/aryan0416/resume-parser-ai/

---

## Features
- **Semantic Match Score (BERT Embeddings)**  
  Computes JD–Resume compatibility using Sentence Transformers (not just keyword overlap).

- **ATS Skill Checklist (Skill Database Matching)**  
  Extracts relevant skills from JD and classifies them into:
  - Found  
  - Weak (mentioned once)  
  - Missing  

- **Resume Section Parsing**  
  Detects sections such as Summary, Skills, Experience, Projects, Education.

- **Section-wise Scoring**  
  Measures alignment by section to pinpoint weaknesses.

- **Action Plan Suggestions**  
  Provides prioritized fix order and exact guidance on where to add missing skills.

- **Bullet Review**  
  Flags weak resume bullets lacking:
  - Action verbs  
  - Metrics  
  - Specificity  

---

## Tech Stack
- **Frontend/UI:** Streamlit + Custom CSS  
- **Semantic Matching:** Sentence Transformers (`all-MiniLM-L6-v2`)  
- **Similarity Metric:** Cosine Similarity  
- **Skill Extraction:** Skill database + regex matching  
- **PDF Parsing:** `pypdf`  
- **Libraries:** NumPy, scikit-learn  

---

## Project Structure
```bash
resume-parser-ai/
│── app.py
│── skills_db.py
│── requirements.txt
│── README.md
│── assets/
````

---

## Installation (Local Setup)

### 1) Clone Repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2) Create Virtual Environment

```bash
python -m venv venv
```

### 3) Activate Virtual Environment

**Windows (PowerShell)**

```powershell
.\venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

### 4) Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 5) Run the App

```bash
streamlit run app.py
```

---

## Deployment (Streamlit Cloud)

1. Push code to GitHub
2. Go to Streamlit Cloud → New App
3. Select repository + branch
4. Set main file as `app.py`
5. Deploy

---

## Screenshots

(Add your screenshots here)

Example:

```md
![Dashboard](assets/01_dashboard.png)
![ATS Checklist](assets/02_ats_checklist.png)
![Action Plan](assets/03_action_plan.png)
![Bullet Review](assets/04_bullet_review.png)
```

---

## Author

**Aryan Dahiya**

* LinkedIn: [https://linkedin.com/in/](https://linkedin.com/in/)<your-profile>
* GitHub: [https://github.com/](https://github.com/)<your-username>

---

## License

This project is licensed under the **MIT License**.

```

If you want, I can also generate a **shorter README** version (minimal but premium) for recruiters + add “How it works” section with architecture diagram text.
```
