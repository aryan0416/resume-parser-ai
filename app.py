import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Function to extract text from PDF ---
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# --- Main App Logic ---
st.set_page_config(page_title="AI Resume Matcher", page_icon="📄")

st.title("📄 AI Resume Parser & Matcher")
st.write("Upload your resume and paste the job description to see how well they match.")

# 1. Inputs
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

with col2:
    job_description = st.text_area("Paste Job Description", height=200)

# 2. Logic Button
if st.button("Analyze Resume"):
    if uploaded_file and job_description:
        with st.spinner("Analyzing..."):
            try:
                # Extract text
                resume_text = extract_text_from_pdf(uploaded_file)
                
                # Combine for vectorization
                documents = [job_description, resume_text]
                
                # Vectorize (Convert text to numbers)
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(documents)
                
                # Calculate Cosine Similarity
                match_percentage = cosine_similarity(tfidf_matrix)[0][1] * 100
                
                # Display Result
                st.divider()
                st.subheader("Analysis Result")
                
                if match_percentage >= 75:
                    st.success(f"**Great Match!** Score: {match_percentage:.2f}%")
                elif match_percentage >= 50:
                    st.warning(f"**Moderate Match.** Score: {match_percentage:.2f}%")
                else:
                    st.error(f"**Low Match.** Score: {match_percentage:.2f}%")
                    
                st.info("Tip: Try adding keywords from the job description into your resume to improve the score.")
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload a PDF and paste a job description.")