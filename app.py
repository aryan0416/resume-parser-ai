import streamlit as st
import PdfReader
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="ResumeAI Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_bert():
    return SentenceTransformer("all-MiniLM-L6-v2")

bert_model = load_bert()

# ---------------- PREMIUM CSS (NO EMOJIS) ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Background */
.stApp{
    background:
        radial-gradient(circle at 12% 18%, rgba(96,165,250,0.20), transparent 42%),
        radial-gradient(circle at 80% 24%, rgba(52,211,153,0.18), transparent 44%),
        linear-gradient(120deg, #0b1220, #111a2e);
    color:#eaf0ff;
}
#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header{visibility:hidden;}

.block-container{padding-top:2rem;padding-bottom:2rem;}

.hero{text-align:center;margin-bottom:26px;}
.hero h1{
    font-size:58px;font-weight:850;letter-spacing:-1.4px;margin:0;
    background:linear-gradient(90deg,#60a5fa,#34d399);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.hero p{margin:10px 0 0;font-size:16px;color:rgba(255,255,255,0.70);}

.card{
    background:rgba(255,255,255,0.06);
    border:1px solid rgba(255,255,255,0.12);
    border-radius:22px;
    box-shadow:0 18px 42px rgba(0,0,0,0.35);
    backdrop-filter:blur(14px);
    -webkit-backdrop-filter:blur(14px);
    padding:22px;
}
.section-title{font-size:18px;font-weight:900;margin:0 0 6px;}
.section-sub{margin:0 0 12px;color:rgba(255,255,255,0.65);font-size:13.5px;}

.stTextArea textarea{
    border-radius:18px!important;
    padding:16px!important;
    font-size:14.5px!important;
    line-height:1.55!important;
    background:rgba(255,255,255,0.06)!important;
    border:1px solid rgba(255,255,255,0.12)!important;
    color:rgba(255,255,255,0.92)!important;
}
section[data-testid="stFileUploaderDropzone"]{
    border-radius:18px!important;
    border:1px dashed rgba(255,255,255,0.22)!important;
    background:rgba(255,255,255,0.05)!important;
    padding:12px!important;
}
.divider{height:1px;background:rgba(255,255,255,0.12);margin:18px 0 22px;}

.center-action{display:flex;justify-content:center;margin-top:6px;margin-bottom:14px;}
.stButton>button{
    width:340px!important;border:none;border-radius:16px;
    padding:12px 18px;font-size:16px;font-weight:900;
    letter-spacing:0.2px;color:#071021;
    background:linear-gradient(90deg,#60a5fa,#34d399);
    box-shadow:0 16px 40px rgba(0,0,0,0.36);
    transition:transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton>button:hover{transform:translateY(-1px);box-shadow:0 20px 50px rgba(0,0,0,0.42);}

.stTabs [data-baseweb="tab-list"]{gap:10px;}
.stTabs [data-baseweb="tab"]{
    height:46px;border-radius:14px;
    background:rgba(255,255,255,0.06);
    border:1px solid rgba(255,255,255,0.10);
    color:rgba(255,255,255,0.78);
    padding:0 16px;font-weight:900;
}
.stTabs [aria-selected="true"]{
    background:rgba(255,255,255,0.12);
    color:rgba(255,255,255,0.95);
    border:1px solid rgba(255,255,255,0.18);
}

.kpi{
    display:grid;
    grid-template-columns: 1fr 1fr 1fr 1fr;
    gap:14px;
}
.kpi-card{
    background:rgba(255,255,255,0.06);
    border:1px solid rgba(255,255,255,0.12);
    border-radius:18px;
    padding:16px;
}
.kpi-card h3{margin:0;font-size:13px;color:rgba(255,255,255,0.70);font-weight:800;}
.kpi-card .v{margin:8px 0 0;font-size:28px;font-weight:950;letter-spacing:-0.6px;}
.kpi-card .s{margin:6px 0 0;font-size:12px;color:rgba(255,255,255,0.62);font-weight:700;}

.badge{
    display:inline-flex;
    padding:6px 10px;
    border-radius:999px;
    font-weight:900;
    font-size:12px;
    border:1px solid rgba(255,255,255,0.14);
    background:rgba(255,255,255,0.06);
    color:rgba(255,255,255,0.86);
}
.skill-grid{
    display:grid;
    grid-template-columns: 1fr 1fr;
    gap:12px;
}
.skill-row{
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:12px 14px;
    border-radius:14px;
    border:1px solid rgba(255,255,255,0.12);
    background:rgba(255,255,255,0.06);
}

.footer{text-align:center;margin-top:40px;color:rgba(255,255,255,0.55);font-size:13px;}
</style>
""", unsafe_allow_html=True)

# ---------------- HELPERS ----------------
def extract_text_from_pdf(file) -> str:
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += "\n" + (page.extract_text() or "")
    return text.strip()

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s+#.%]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def semantic_score(a: str, b: str) -> float:
    ea = bert_model.encode([a])
    eb = bert_model.encode([b])
    return float(cosine_similarity(ea, eb)[0][0] * 100)

def score_color(score: float) -> str:
    if score >= 80: return "#34d399"
    if score >= 60: return "#fbbf24"
    return "#fb7185"

def label(score: float) -> str:
    if score >= 85: return "Excellent Match"
    if score >= 70: return "Strong Match"
    if score >= 55: return "Average Match"
    return "Low Match"

# --- Section parsing ---
SECTION_ALIASES = {
    "summary": ["summary", "profile", "objective", "about"],
    "skills": ["skills", "technical skills", "key skills", "core skills"],
    "experience": ["experience", "work experience", "employment", "internship", "internships"],
    "projects": ["projects", "project", "academic projects"],
    "education": ["education", "academics", "qualification", "qualifications"],
}

def normalize_heading(line: str) -> str:
    return re.sub(r"[^a-z\s]", "", line.lower()).strip()

def detect_heading(line: str):
    x = normalize_heading(line)
    for section, aliases in SECTION_ALIASES.items():
        for a in aliases:
            if x == a:
                return section
    return None

def split_sections(resume_text: str) -> dict:
    lines = [l.strip() for l in resume_text.splitlines() if l.strip()]
    sections = {k: "" for k in SECTION_ALIASES.keys()}
    current = None
    buffer = []

    for line in lines:
        maybe = detect_heading(line)
        if maybe:
            if current and buffer:
                sections[current] += "\n".join(buffer).strip() + "\n"
            current = maybe
            buffer = []
        else:
            buffer.append(line)

    if current and buffer:
        sections[current] += "\n".join(buffer).strip()

    # fallback if nothing detected: keep full text as summary bucket
    if all(len(v.strip()) == 0 for v in sections.values()):
        sections["summary"] = resume_text

    return sections

# --- JD keyword + ATS checklist ---
GENERIC_BAD_TERMS = {
    "job","jobs","job description","india","jobs india","private","limited","private limited",
    "opportunity","world","real world","hands","role","based stipend","stipend","based performance",
    "company","candidate","candidates","developer","development","software","software development",
    "projects","development projects","apply","hiring","location","salary","joining","interview"
}

# (Optional) add more terms based on your JDs

def looks_like_skill(term: str) -> bool:
    """
    Keep skill-looking terms:
    - contains tech chars: +, ., -, /
    - contains common tech words
    - is single/short phrase (<=3 words)
    """
    t = term.lower().strip()

    if t in GENERIC_BAD_TERMS:
        return False

    if len(t) < 2:
        return False

    # too long phrases are usually HR fluff
    if len(t.split()) > 3:
        return False

    tech_hints = [
        "python","java","c++","c","javascript","typescript","sql","mongodb","mysql","postgres",
        "react","node","express","django","flask","spring","html","css","tailwind",
        "ml","ai","nlp","cnn","rnn","transformer","tensorflow","pytorch","sklearn",
        "git","github","docker","kubernetes","linux","api","rest","graphql",
        "aws","azure","gcp","cloud","devops","ci/cd","firebase"
    ]

    # direct match with hints
    if any(h in t for h in tech_hints):
        return True

    # if contains symbols common in tech terms
    if any(ch in t for ch in ["+", ".", "/", "-", "_"]):
        return True

    # must contain at least one alphabet
    if not any(c.isalpha() for c in t):
        return False

    # avoid generic business/HR words
    bad_words = {"opportunity","hands","world","role","based","performance"}
    if any(w in t.split() for w in bad_words):
        return False

    # fallback: accept 1-word tokens that look technical
    if len(t.split()) == 1 and len(t) <= 18:
        return True

    return False


def jd_keywords(jd_text: str, top_k=60):
    jd = clean_text(jd_text)
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=4000)
    X = vec.fit_transform([jd]).toarray().flatten()
    vocab = np.array(vec.get_feature_names_out())
    idx = np.argsort(X)[::-1]

    out = []
    for term in vocab[idx]:
        term = term.strip().lower()
        if looks_like_skill(term):
            out.append(term)
        if len(out) >= top_k:
            break
    return out


def ats_checklist(jd_text: str, resume_text: str, top_k=26):
    keys = jd_keywords(jd_text, top_k=top_k * 3)  # collect more then trim
    res = clean_text(resume_text)

    found, missing = [], []
    for k in keys:
        if res.count(k) > 0:
            found.append(k)
        else:
            missing.append(k)

    # limit output
    return found[:top_k], missing[:top_k]


# --- Bullet quality analysis ---
ACTION_VERBS = {
    "developed","built","created","designed","implemented","improved","optimized","led",
    "managed","delivered","deployed","engineered","integrated","automated","analyzed",
    "trained","evaluated","researched","tested","debugged","maintained","scaled"
}

def split_bullets(text: str):
    raw = re.split(r"\n|•|- |– ", text)
    bullets = [b.strip() for b in raw if len(b.strip()) >= 18]
    return bullets

def bullet_quality(bullet: str):
    b = bullet.strip()
    low = b.lower()

    has_action = any(low.startswith(v + " ") for v in ACTION_VERBS)
    has_metric = bool(re.search(r"\b\d+(\.\d+)?\s?(%|ms|s|sec|seconds|mins|min|hrs|hours|x|times|k|m|million|billion)?\b", low))
    length_ok = len(b) >= 40

    issues = []
    if not has_action:
        issues.append("No strong action verb at start")
    if not has_metric:
        issues.append("No measurable metric (%, time, scale, count)")
    if not length_ok:
        issues.append("Too short / generic bullet")

    return issues

# --- Section-wise similarity ---
def section_scores(jd_text: str, sections: dict):
    scores = {}
    for sec, txt in sections.items():
        if txt.strip():
            scores[sec] = semantic_score(jd_text, txt)
        else:
            scores[sec] = 0.0
    return scores

def overall_weighted(section_score_map: dict):
    # weights tuned for typical ATS relevance
    w = {"skills": 0.35, "experience": 0.30, "projects": 0.22, "summary": 0.08, "education": 0.05}
    total = 0.0
    for k, wt in w.items():
        total += wt * section_score_map.get(k, 0.0)
    return total

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
  <h1>ResumeAI Pro</h1>
  <p>Section-aware ATS analysis, skill checklist, and actionable resume improvements</p>
</div>
""", unsafe_allow_html=True)

# ---------------- INPUT AREA ----------------
c1, c2 = st.columns([1.25, 1], gap="large")
with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Job Description</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Paste the complete job description below.</div>', unsafe_allow_html=True)
    jd = st.text_area("Job Description", height=290, label_visibility="collapsed",
                      placeholder="Paste job description here...")
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Resume Upload</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Upload your resume in PDF format.</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Resume PDF", type=["pdf"], label_visibility="collapsed")
    if uploaded:
        st.markdown("<div class='section-sub'>File attached successfully.</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

st.markdown("<div class='center-action'>", unsafe_allow_html=True)
analyze = st.button("Analyze Compatibility")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- RESULTS ----------------
if analyze:
    if not uploaded or not jd.strip():
        st.warning("Please upload a resume PDF and paste the job description.")
    else:
        with st.spinner("Analyzing..."):
            resume_text = extract_text_from_pdf(uploaded)
            sections = split_sections(resume_text)

            sec_scores = section_scores(jd, sections)
            overall = overall_weighted(sec_scores)
            overall_c = score_color(overall)

            found_skills, missing_skills = ats_checklist(jd, resume_text)

            # bullet checks only for experience/projects
            exp_bullets = split_bullets(sections.get("experience", ""))
            proj_bullets = split_bullets(sections.get("projects", ""))

            weak_bullets = []
            for b in (exp_bullets + proj_bullets):
                issues = bullet_quality(b)
                if issues:
                    weak_bullets.append((b, issues))

        tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "ATS Checklist", "Action Plan", "Bullet Review"])

        # ---------------- DASHBOARD ----------------
        with tab1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Overall Dashboard</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">A complete breakdown of alignment by section.</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="kpi">
                <div class="kpi-card">
                    <h3>Overall Score</h3>
                    <div class="v" style="color:{overall_c};">{overall:.1f}%</div>
                    <div class="s">{label(overall)}</div>
                </div>
                <div class="kpi-card">
                    <h3>Skills Score</h3>
                    <div class="v">{sec_scores["skills"]:.1f}%</div>
                    <div class="s">Keywords + relevance</div>
                </div>
                <div class="kpi-card">
                    <h3>Experience Score</h3>
                    <div class="v">{sec_scores["experience"]:.1f}%</div>
                    <div class="s">Role alignment</div>
                </div>
                <div class="kpi-card">
                    <h3>Projects Score</h3>
                    <div class="v">{sec_scores["projects"]:.1f}%</div>
                    <div class="s">Impact + tools</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

            # Section state
            s1, s2, s3 = st.columns(3, gap="large")
            with s1:
                st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
                st.markdown("<div class='section-title'>Detected Sections</div>", unsafe_allow_html=True)
                for k in ["summary","skills","experience","projects","education"]:
                    status = "Present" if sections.get(k, "").strip() else "Not detected"
                    st.markdown(f"<div class='skill-row'><span style='font-weight:900'>{k.title()}</span><span class='badge'>{status}</span></div>",
                                unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with s2:
                st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
                st.markdown("<div class='section-title'>Skill Coverage</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='section-sub'>Found: {len(found_skills)} | Missing: {len(missing_skills)}</div>", unsafe_allow_html=True)
                st.progress(min(len(found_skills) / max(1, len(found_skills) + len(missing_skills)), 1.0))
                st.markdown("<div class='section-sub'>This indicates how many high-value JD signals appear in your resume.</div>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with s3:
                st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
                st.markdown("<div class='section-title'>Resume Quality Flags</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='section-sub'>Weak bullets detected: {len(weak_bullets)}</div>", unsafe_allow_html=True)
                st.markdown("<div class='section-sub'>Weak bullets are generic lines lacking action verbs or measurable outcomes.</div>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # ---------------- ATS CHECKLIST ----------------
        with tab2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">ATS Checklist</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Key signals extracted from JD and checked against your resume.</div>', unsafe_allow_html=True)

            a, b = st.columns(2, gap="large")
            with a:
                st.markdown("<div class='section-title'>Found Skills</div>", unsafe_allow_html=True)
                if found_skills:
                    for k in found_skills[:30]:
                        st.markdown(f"<div class='skill-row'><span style='font-weight:900'>{k}</span><span class='badge'>Found</span></div>",
                                    unsafe_allow_html=True)
                else:
                    st.markdown("<div class='section-sub'>No JD-aligned skills found.</div>", unsafe_allow_html=True)

            with b:
                st.markdown("<div class='section-title'>Missing Skills</div>", unsafe_allow_html=True)
                if missing_skills:
                    for k in missing_skills[:30]:
                        st.markdown(f"<div class='skill-row'><span style='font-weight:900'>{k}</span><span class='badge'>Missing</span></div>",
                                    unsafe_allow_html=True)
                else:
                    st.markdown("<div class='section-sub'>No missing skills detected.</div>", unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # ---------------- ACTION PLAN ----------------
        with tab3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Action Plan</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Exact changes to increase score quickly.</div>', unsafe_allow_html=True)

            # Identify weakest sections
            ranked = sorted(sec_scores.items(), key=lambda x: x[1])
            weakest = [k for k, _ in ranked[:2]]

            st.markdown("<div class='section-title'>Priority Fix Order</div>", unsafe_allow_html=True)
            st.write(f"1) {weakest[0].title()}  2) {weakest[1].title()}  3) Skills keywords alignment")

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

            # Specific suggestions
            st.markdown("<div class='section-title'>What to add and where</div>", unsafe_allow_html=True)

            if missing_skills:
                top_missing = missing_skills[:10]
                st.write("Add these missing signals in your resume naturally:")
                st.write(", ".join(top_missing))

                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

                st.markdown("<div class='section-title'>Placement recommendations</div>", unsafe_allow_html=True)
                st.write("- Add 6–10 of them inside the Skills section.")
                st.write("- Add 3–5 of them inside Projects bullet points.")
                st.write("- Add 2–3 inside Experience bullet points (only if you truly used them).")

            else:
                st.write("No major missing skills detected. Focus on bullet impact and measurable metrics.")

            # Metrics suggestion based on bullet review
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Impact improvements</div>", unsafe_allow_html=True)
            st.write("- Ensure every major project has: tools + dataset/scale + metric.")
            st.write("- Use numbers: accuracy %, latency ms, users, time saved, cost reduced, throughput.")

            st.markdown('</div>', unsafe_allow_html=True)

        # ---------------- BULLET REVIEW ----------------
        with tab4:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Bullet Review</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Detects generic bullets and explains exactly why they are weak.</div>', unsafe_allow_html=True)

            if not weak_bullets:
                st.write("No weak bullets detected in projects/experience. Bullet quality is strong.")
            else:
                for i, (b, issues) in enumerate(weak_bullets[:10], start=1):
                    st.markdown(f"<div class='kpi-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-weight:950;'>Bullet {i}</div>", unsafe_allow_html=True)
                    st.write(b)
                    st.markdown("<div class='section-sub' style='margin-top:8px;'>Issues detected:</div>", unsafe_allow_html=True)
                    for it in issues:
                        st.write(f"- {it}")
                    st.markdown("<div class='section-sub' style='margin-top:8px;'>Suggested pattern:</div>", unsafe_allow_html=True)
                    st.write("Action verb + what you built + tools used + scale/dataset + measurable result")
                    st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
Designed & Developed by Aryan Dahiya | © 2026
</div>
""", unsafe_allow_html=True)


