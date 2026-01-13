# ResumeAI Pro — ATS Resume Screening & Optimization Tool

ResumeAI Pro is an ATS-inspired resume screening web app that compares a candidate’s resume against a job description and generates actionable improvement insights.  
It combines semantic similarity matching with a skill-database checklist to help candidates tailor their resumes to job requirements.

---

## Live Demo
🔗 **App:** https://<your-streamlit-app-link>  
🔗 **GitHub:** https://github.com/<your-username>/<repo-name>

---

## Features
- **Semantic Match Score (BERT Embeddings):**
  Computes JD–Resume compatibility using Sentence Transformers (not just keyword overlap).
- **ATS Skill Checklist (Skill Database Matching):**
  Detects JD required skills and classifies them into:
  - Found
  - Weak (mentioned once)
  - Missing
- **Resume Section Parsing:**
  Automatically detects sections like:
  - Summary
  - Skills
  - Experience
  - Projects
  - Education
- **Section-wise Scoring:**
  Individual compatibility scores for each section to identify weak areas.
- **Action Plan:**
  Gives a prioritized fix order and exact guidance on where to add missing skills.
- **Bullet Review:**
  Flags weak bullets that lack:
  - Action verbs
  - Metrics
  - Depth / specificity

---

## Tech Stack
- **Frontend/UI:** Streamlit + Custom CSS
- **NLP/Semantic Matching:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Similarity Metric:** Cosine Similarity
- **Skill Extraction:** Skill database + regex matching
- **PDF Parsing:** `pypdf`
- **Other:** NumPy, scikit-learn

---

## Project Structure
resume-parser-ai/
│── app.py
│── skills_db.py
│── requirements.txt
│── README.md

yaml
Copy code

---

## Installation (Local Setup)

### 1) Clone Repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
2) Create Virtual Environment
bash
Copy code
python -m venv venv
Activate it:

Windows (PowerShell)

powershell
Copy code
.\venv\Scripts\activate
Mac/Linux

bash
Copy code
source venv/bin/activate
3) Install Dependencies
bash
Copy code
python -m pip install -r requirements.txt
4) Run Streamlit App
bash
Copy code
streamlit run app.py
Deployment (Streamlit Cloud)
Push code to GitHub

Go to Streamlit Cloud → "New App"

Select repo + branch

Set main file: app.py

Deploy

Screenshots
(Add your screenshots here)

Example:

Dashboard

ATS checklist

Action plan

Bullet review

Future Improvements
Report export (PDF)

Resume bullet rewriting (STAR format)

Skill categories & weighted scoring

Multi-resume comparison

Support DOCX resumes

Author
Aryan Dahiya

LinkedIn: https://linkedin.com/in/<your-profile>

GitHub: https://github.com/<your-username>

