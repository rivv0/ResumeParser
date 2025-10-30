import streamlit as st
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import requests
import re

# Page config
st.set_page_config(
    page_title="ML Resume Matcher",
    page_icon="🤖",
    layout="wide"
)

# Cache the model loading
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Your skills database
skills_db = [
    'python', 'java', 'javascript', 'typescript', 'react', 'angular', 'vue', 'node.js',
    'sql', 'mysql', 'postgresql', 'mongodb', 'aws', 'azure', 'docker', 'kubernetes',
    'sales', 'marketing', 'design', 'cooking', 'finance', 'accounting', 'management'
]

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None
    return text

def extract_skills(text):
    found_skills = []
    text_lower = text.lower()
    for skill in skills_db:
        if skill in text_lower:
            found_skills.append(skill)
    return found_skills

# Main app
st.title("🤖 ML Resume Matcher")
st.markdown("Upload your resume and get AI-powered job matching!")

# Load model
with st.spinner("Loading AI model..."):
    model = load_model()

# File upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Analyzing resume..."):
        # Extract text
        resume_text = extract_text_from_pdf(uploaded_file)
        
        if resume_text:
            # Extract skills
            skills = extract_skills(resume_text)
            
            # Sample job matching
            job_desc = "We need a Python developer with React and AWS experience for our startup"
            
            # Calculate similarity
            resume_embedding = model.encode(resume_text[:1000])
            job_embedding = model.encode(job_desc)
            similarity = cosine_similarity([resume_embedding], [job_embedding])[0][0]
            
            score = min((similarity + 0.4) * 100, 95)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Analysis Results")
                st.metric("Overall Match Score", f"{score:.1f}%")
                st.metric("Semantic Similarity", f"{similarity*100:.1f}%")
                st.metric("Skills Found", len(skills))
                
            with col2:
                st.subheader("🎯 Skills Detected")
                if skills:
                    for skill in skills[:10]:
                        st.badge(skill)
                else:
                    st.info("No specific skills detected")
            
            st.subheader("📄 Resume Preview")
            st.text_area("Extracted Text", resume_text[:500] + "...", height=200)
            
            st.subheader("💼 Sample Job Match")
            st.info(f"**Position:** Python Developer\n**Score:** {score:.1f}%\n**Description:** {job_desc}")
        
        else:
            st.error("Could not extract text from PDF. Please try another file.")

st.markdown("---")
st.markdown("Built with Streamlit • Powered by Sentence Transformers")