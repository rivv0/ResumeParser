import os
import pandas as pd
import re
from PyPDF2 import PdfReader
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import Counter
import json

# Load spaCy model for NLP (install: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Please install spaCy model: python -m spacy download en_core_web_sm")

class EnhancedResumeParser:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def extract_text_pypdf2(self, pdf_path):
        """Extract text from PDF"""
        text = ""
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
        return text
    
    def extract_email(self, text):
        """Extract email addresses"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        return emails[0] if emails else None
    
    def extract_phone(self, text):
        """Extract phone numbers"""
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phones = re.findall(phone_pattern, text)
        return phones[0] if phones else None
    
    def extract_education(self, text):
        """Extract education information"""
        education_keywords = [
            'Bachelor', 'Master', 'PhD', 'B.Tech', 'M.Tech', 'MBA', 'B.S', 'M.S',
            'BSc', 'MSc', 'BE', 'ME', 'Diploma', 'Associate', 'Doctorate'
        ]
        
        education = []
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if any(keyword.lower() in line.lower() for keyword in education_keywords):
                education.append(line.strip())
        
        return ' | '.join(education[:3]) if education else None
    
    def extract_skills(self, text):
        """Extract technical skills using NLP and keyword matching"""
        # Common technical skills database
        skill_keywords = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin', 'go', 'rust', 'typescript'],
            'web': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring', 'asp.net'],
            'data': ['sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'redis', 'cassandra', 'elasticsearch'],
            'ml_ai': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'nlp', 'computer vision'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible'],
            'tools': ['git', 'jira', 'agile', 'scrum', 'linux', 'bash', 'powershell']
        }
        
        text_lower = text.lower()
        found_skills = []
        
        for category, skills in skill_keywords.items():
            for skill in skills:
                if skill in text_lower:
                    found_skills.append(skill)
        
        return list(set(found_skills))
    
    def extract_experience_years(self, text):
        """Extract years of experience"""
        # Pattern: "X years of experience" or "X+ years"
        exp_patterns = [
            r'(\d+)\+?\s*years?\s*of\s*experience',
            r'experience\s*[:\-]?\s*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*experience'
        ]
        
        for pattern in exp_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return int(match.group(1))
        
        return None
    
    def extract_certifications(self, text):
        """Extract certifications"""
        cert_keywords = ['certified', 'certification', 'certificate', 'credential']
        certifications = []
        
        lines = text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in cert_keywords):
                certifications.append(line.strip())
        
        return ' | '.join(certifications[:3]) if certifications else None
    
    def generate_embeddings(self, text):
        """Generate semantic embeddings for the resume"""
        if not text or len(text.strip()) < 10:
            return None
        
        # Truncate text if too long (max 512 tokens for most models)
        text_truncated = text[:5000]
        embedding = self.embedding_model.encode(text_truncated)
        return embedding
    
    def extract_entities(self, text):
        """Extract named entities using spaCy"""
        doc = nlp(text[:100000])  # Limit text length for processing
        
        entities = {
            'organizations': [],
            'locations': [],
            'persons': []
        }
        
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                entities['organizations'].append(ent.text)
            elif ent.label_ in ['GPE', 'LOC']:
                entities['locations'].append(ent.text)
            elif ent.label_ == 'PERSON':
                entities['persons'].append(ent.text)
        
        return entities
    
    def calculate_text_metrics(self, text):
        """Calculate various text metrics"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        return {
            'word_count': len(words),
            'char_count': len(text),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0
        }
    
    def process_resume(self, pdf_path, category, filename):
        """Process a single resume and extract all features"""
        text = self.extract_text_pypdf2(pdf_path)
        
        if not text or len(text.strip()) < 50:
            return None
        
        # Extract all features
        resume_data = {
            'resume_id': filename.replace('.pdf', ''),
            'category': category,
            'resume_text': text,
            'email': self.extract_email(text),
            'phone': self.extract_phone(text),
            'education': self.extract_education(text),
            'skills': self.extract_skills(text),
            'skills_count': len(self.extract_skills(text)),
            'experience_years': self.extract_experience_years(text),
            'certifications': self.extract_certifications(text),
            'text_metrics': self.calculate_text_metrics(text)
        }
        
        # Add entities
        entities = self.extract_entities(text)
        resume_data['organizations'] = entities['organizations'][:5]
        resume_data['locations'] = entities['locations'][:3]
        
        # Generate embeddings
        resume_data['embedding'] = self.generate_embeddings(text)
        
        return resume_data
    
    def process_all_resumes(self):
        """Process all resumes in the dataset"""
        resume_data_list = []
        categories = os.listdir(self.data_dir)
        
        total_resumes = sum(
            len([f for f in os.listdir(os.path.join(self.data_dir, cat)) 
                 if f.endswith('.pdf')])
            for cat in categories if os.path.isdir(os.path.join(self.data_dir, cat))
        )
        
        print(f"Total resumes found: {total_resumes}")
        processed = 0
        
        for category in categories:
            category_path = os.path.join(self.data_dir, category)
            
            if not os.path.isdir(category_path):
                continue
            
            for filename in os.listdir(category_path):
                if not filename.endswith('.pdf'):
                    continue
                
                pdf_path = os.path.join(category_path, filename)
                resume_data = self.process_resume(pdf_path, category, filename)
                
                if resume_data:
                    resume_data_list.append(resume_data)
                
                processed += 1
                if processed % 50 == 0:
                    print(f"Processed {processed}/{total_resumes} resumes ({processed/total_resumes*100:.1f}%)")
        
        return pd.DataFrame(resume_data_list)


# Usage
if __name__ == "__main__":
    data_dir = "/Users/rivva/Build/jobie/resumeparser/input/data/data"
    
    parser = EnhancedResumeParser(data_dir)
    resume_df = parser.process_all_resumes()
    
    # Save embeddings separately (they're numpy arrays)
    embeddings = np.array([emb for emb in resume_df['embedding'].values if emb is not None])
    np.save('resume_embeddings.npy', embeddings)
    
    # Save the rest to CSV
    resume_df_export = resume_df.drop('embedding', axis=1)
    resume_df_export.to_csv('enhanced_resume_data.csv', index=False)
    
    print("\nProcessing complete!")
    print(f"Total processed resumes: {len(resume_df)}")
    print(f"\nSample data:\n{resume_df.head()}")