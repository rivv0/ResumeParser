import gradio as gr
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import requests
import re

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Your existing skills database (shortened for demo)
skills_db = [
    'python', 'java', 'javascript', 'react', 'node.js', 'sql', 'aws', 'docker',
    'sales', 'marketing', 'design', 'cooking', 'finance', 'accounting'
]

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    except:
        return None
    return text

def extract_skills(text):
    found_skills = []
    text_lower = text.lower()
    for skill in skills_db:
        if skill in text_lower:
            found_skills.append(skill)
    return found_skills

def match_resume(pdf_file):
    if pdf_file is None:
        return "Please upload a PDF file"
    
    # Extract text
    resume_text = extract_text_from_pdf(pdf_file)
    if not resume_text:
        return "Could not extract text from PDF"
    
    # Extract skills
    skills = extract_skills(resume_text)
    
    # Sample job for demo
    job_desc = "We need a Python developer with React and AWS experience"
    
    # Calculate similarity
    resume_embedding = model.encode(resume_text[:1000])
    job_embedding = model.encode(job_desc)
    similarity = cosine_similarity([resume_embedding], [job_embedding])[0][0]
    
    score = (similarity + 0.4) * 100  # Boost score
    
    result = f"""
    **Resume Analysis Complete!**
    
    **Skills Found:** {', '.join(skills[:10]) if skills else 'None detected'}
    
    **Sample Job Match:**
    - Position: Python Developer
    - Match Score: {score:.1f}%
    - Semantic Similarity: {similarity*100:.1f}%
    
    **Resume Preview:** {resume_text[:200]}...
    """
    
    return result

# Create Gradio interface
demo = gr.Interface(
    fn=match_resume,
    inputs=gr.File(label="Upload Resume (PDF)", file_types=[".pdf"]),
    outputs=gr.Markdown(label="Match Results"),
    title="🤖 ML Resume Matcher",
    description="Upload your resume and get AI-powered job matching results!",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()