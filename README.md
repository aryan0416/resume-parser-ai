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

<img width="1891" height="953" alt="image" src="https://github.com/user-attachments/assets/0fdb6514-fb18-405b-8f1e-5fa43c7e7442" />
<img width="1912" height="959" alt="image" src="https://github.com/user-attachments/assets/38812621-63f5-40d6-9e43-d91cca51785c" />
<img width="1902" height="950" alt="image" src="https://github.com/user-attachments/assets/b30c055c-0733-4a23-a92b-d83f9f31cc0b" />
<img width="1757" height="948" alt="image" src="https://github.com/user-attachments/assets/88a2f294-5eab-4828-9684-d3fee79f31ba" />
<img width="1902" height="928" alt="image" src="https://github.com/user-attachments/assets/56a8dedc-eea9-423b-8118-37cf7b451f68" />


---

## Author

**Aryan Dahiya**

* LinkedIn:(https://www.linkedin.com/in/aryan-dahiya/)
* GitHub: https://github.com/aryan0416

---

## License

This project is licensed under the **MIT License**.


