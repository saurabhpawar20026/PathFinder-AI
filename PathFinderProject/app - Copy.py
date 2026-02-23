import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

# --- Page Config ---
st.set_page_config(page_title="PathFinder AI", layout="wide")

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# --- UI Header ---
st.title("🚀 PathFinder AI: Smart Skill-Gap Analyzer")
st.subheader("Placement Coordinator Special Edition")
st.write("Compare your Resume with Job Descriptions to find missing skills.")

# --- Sidebar Inputs ---
st.sidebar.header("Upload Details")
uploaded_file = st.sidebar.file_uploader("Upload your Resume (PDF)", type="pdf")
job_description = st.sidebar.text_area("Paste Job Description (JD) here:", height=300)

if st.sidebar.button("Analyze Now"):
    if uploaded_file is not None and job_description != "":
        # Process Resume
        resume_text = extract_text_from_pdf(uploaded_file)
        
        # Vectorization for Matching
        content = [resume_text, job_description]
        cv = TfidfVectorizer()
        matrix = cv.fit_transform(content)
        similarity_matrix = cosine_similarity(matrix)
        match_percentage = round(similarity_matrix[0][1] * 100, 2)
        
        # Display Results
        st.header(f"Match Score: {match_percentage}%")
        
        if match_percentage >= 70:
            st.success("✅ Strong Match! Your profile fits the JD well.")
        elif match_percentage >= 40:
            st.warning("⚠️ Average Match. You need to add some keywords.")
        else:
            st.error("❌ Low Match. High Skill Gap detected.")

        # Skill Gap Logic (Basic Keyword Comparison)
        jd_keywords = set(job_description.lower().split())
        resume_keywords = set(resume_text.lower().split())
        missing_skills = jd_keywords - resume_keywords
        
        # Filter common filler words (simplified version)
        common_words = {'and', 'the', 'is', 'for', 'to', 'with', 'in', 'of', 'a', 'an'}
        final_missing = [word for word in missing_skills if word.isalpha() and word not in common_words]

        st.subheader("💡 Missing Keywords/Skills in your Resume:")
        st.write(", ".join(final_missing[:15])) # Showing top 15

        # Career Roadmap Recommendation
        st.info("### 🛣️ Quick 30-Day Roadmap")
        st.write("1. **Day 1-10:** Focus on learning: " + ", ".join(final_missing[:3]))
        st.write("2. **Day 11-20:** Build a small project using these skills.")
        st.write("3. **Day 21-30:** Update Resume & apply on LinkedIn.")

    else:
        st.error("Please upload a resume and paste a JD first!")