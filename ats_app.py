# ATS Resume Checker - Streamlit App
# Single-file Streamlit application implementing an end-to-end ATS resume checker
# Features:
# - Upload resume (PDF/DOCX/TXT)
# - Paste or upload Job Description
# - Choose matching method: TF-IDF or SBERT (semantic)
# - Show match score, missing keywords, keyword density
# - Offer suggested bullet points for missing skills
# - Downloadable report (TXT)

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import io
import os
import tempfile
import re

# Optional libraries for parsing
try:
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    import docx2txt
except Exception:
    docx2txt = None

# Basic skill-list (extendable). You can replace or upload your own CSV of skills.
DEFAULT_SKILLS = [
    "python","sql","machine learning","deep learning","tensorflow","pytorch",
    "nlp","computer vision","data analysis","tableau","power bi","excel",
    "git","docker","aws","azure","rest api","javascript","react","node.js",
]

st.set_page_config(page_title="ATS Resume Checker", layout="wide")
st.title("ATS Resume Checker — Full Web App (Streamlit)")

# Sidebar: options
st.sidebar.header("Options")
method = st.sidebar.selectbox("Similarity Method", ["TF-IDF", "SBERT (semantic)"])
show_bullets = st.sidebar.checkbox("Generate suggested bullets for missing skills", value=True)
allow_download = st.sidebar.checkbox("Enable report download", value=True)

# Input: Job Description
st.subheader("Job Description")
job_text = st.text_area("Paste job description here (or upload a .txt file)", height=200)
uploaded_jd = st.file_uploader("Or upload a Job Description file (.txt)", type=["txt"] )
if uploaded_jd is not None and uploaded_jd.type == "text/plain":
    job_text = uploaded_jd.getvalue().decode("utf-8")

# Upload resume
st.subheader("Upload Resume")
resume_file = st.file_uploader("Upload resume (pdf, docx, txt)", type=["pdf","docx","txt"] )

# Optional: custom skills list upload
st.sidebar.markdown("---")
skills_upload = st.sidebar.file_uploader("Upload custom skills list (one per line .txt)", type=["txt"] )
if skills_upload is not None:
    skills_text = skills_upload.getvalue().decode("utf-8")
    skills_list = [s.strip().lower() for s in skills_text.splitlines() if s.strip()]
else:
    skills_list = DEFAULT_SKILLS

# Utility: extract text from resume
def extract_text_from_pdf(file_bytes):
    if pdfplumber is None:
        return ""  # pdfplumber not installed
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n".join(pages)
    except Exception as e:
        return ""

def extract_text_from_docx(file_bytes):
    if docx2txt is None:
        return ""
    try:
        # write to temp file then use docx2txt
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        text = docx2txt.process(tmp_path)
        os.remove(tmp_path)
        return text
    except Exception as e:
        return ""

def extract_text_from_txt(file_bytes):
    try:
        return file_bytes.decode('utf-8', errors='ignore')
    except Exception:
        return ""

# Basic preprocessing
def preprocess(text):
    text = text or ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\.\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Simple keyword extraction from JD using provided skills_list
def extract_skills_from_text(text, skills_vocab):
    text_low = text.lower()
    found = set()
    for skill in skills_vocab:
        if skill in text_low:
            found.add(skill)
    return sorted(found)

# Generate suggested bullets for missing skills
BULLET_TEMPLATES = [
    "Worked on {skill} to {action} with measurable outcome.",
    "Implemented {skill} in projects resulting in improved performance/efficiency.",
    "Developed production-ready {skill} pipeline and integrated with team workflows.",
]

def generate_bullets(missing_skills, n=2):
    bullets = []
    for s in missing_skills:
        for t in BULLET_TEMPLATES[:n]:
            bullets.append(t.format(skill=s, action="develop models / build dashboards / automate processes"))
    return bullets

# Main processing
if st.button("Run ATS Check"):
    if not job_text:
        st.error("Please paste or upload the Job Description.")
    elif resume_file is None:
        st.error("Please upload a resume file.")
    else:
        # extract resume text
        b = resume_file.getvalue()
        ext = resume_file.type
        resume_text = ""
        if resume_file.name.lower().endswith('.pdf'):
            resume_text = extract_text_from_pdf(b)
        elif resume_file.name.lower().endswith('.docx'):
            resume_text = extract_text_from_docx(b)
        else:
            resume_text = extract_text_from_txt(b)

        if not resume_text:
            st.warning("Could not extract text cleanly from resume. The file may be scanned or encrypted.")

        # preprocess
        resume_p = preprocess(resume_text)
        jd_p = preprocess(job_text)

        # skill extraction
        jd_skills = extract_skills_from_text(jd_p, skills_list)
        resume_skills = extract_skills_from_text(resume_p, skills_list)

        # similarity
        if method == "TF-IDF":
            vec = TfidfVectorizer(ngram_range=(1,2)).fit([resume_p, jd_p])
            X = vec.transform([resume_p, jd_p])
            score = float(cosine_similarity(X[0], X[1])[0,0])
        else:
            # SBERT
            try:
                model = SentenceTransformer('all-MiniLM-L6-v2')
                emb = model.encode([resume_p, jd_p])
                score = float(cosine_similarity([emb[0]], [emb[1]])[0,0])
            except Exception as e:
                st.error("SentenceTransformer model not available. Install 'sentence-transformers' and try again.")
                score = 0.0

        match_percent = round(score * 100, 2)

        # missing skills
        missing = [s for s in jd_skills if s not in resume_skills]

        # keyword density (simple counts)
        def keyword_density(text, keywords):
            words = text.split()
            n = len(words) if len(words)>0 else 1
            densities = {}
            for k in keywords:
                count = text.count(k)
                densities[k] = round((count/n)*100, 4)
            return densities

        densities = keyword_density(resume_p, jd_skills)

        # display
        c1, c2 = st.columns([1,1])
        with c1:
            st.metric("Match Score", f"{match_percent}%")
            st.write("**Skills found in JD**")
            st.write(jd_skills)
            st.write("**Skills found in Resume**")
            st.write(resume_skills)
        with c2:
            st.write("**Missing Skills (from JD)**")
            if missing:
                st.write(missing)
            else:
                st.write("None — Good match for skills list provided.")
            st.write("**Keyword density in resume (for JD skills)**")
            st.write(densities)

        # suggested bullets
        if show_bullets and missing:
            st.subheader("Suggested Bullet Points to Add")
            bullets = generate_bullets(missing, n=2)
            for b in bullets:
                st.write("- ", b)

        # prepare report
        report_lines = [
            f"Match Score: {match_percent}%", "\n",
            "Skills in JD:", ", ".join(jd_skills), "\n",
            "Skills in Resume:", ", ".join(resume_skills), "\n",
            "Missing Skills:", ", ".join(missing), "\n",
            "Keyword densities:", str(densities), "\n",
        ]
        report_content = "\n".join([str(x) for x in report_lines])

        if allow_download:
            st.download_button("Download Report (TXT)", data=report_content, file_name="ats_report.txt")

        st.success("ATS Check completed.")

# Footer
st.markdown("---")
st.caption("Built with Streamlit — extend skills list, templates, and parsing as needed.")
