import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from PyPDF2 import PdfReader
import spacy
import re
from collections import Counter
import tempfile

# Page configuration
st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 48px;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
    }
    .job-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-left: 5px solid #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    resume_df = pd.read_csv('enhanced_resume_data.csv')
    embeddings = np.load('resume_embeddings.npy')
    return resume_df, embeddings

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Initialize
try:
    resume_df, embeddings = load_data()
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ Data files not found! Please run the enhanced parser first.")
    st.stop()

# Load spaCy model
@st.cache_resource
def load_nlp_model():
    try:
        return spacy.load("en_core_web_sm")
    except:
        st.error("Please install spaCy model: python -m spacy download en_core_web_sm")
        return None

nlp = load_nlp_model()

# Resume parsing functions
def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF"""
    text = ""
    try:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

def extract_skills_from_text(text):
    """Extract skills from resume text with comprehensive skill database"""
    skill_keywords = {
        # TECHNOLOGY & IT
        'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin', 'go', 'rust', 'typescript', 'programming', 'coding', 'software development'],
        'web': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring', 'asp.net', 'web development', 'frontend', 'backend'],
        'data': ['sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'redis', 'cassandra', 'elasticsearch', 'database', 'data analysis'],
        'ml_ai': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'nlp', 'computer vision', 'artificial intelligence', 'data science'],
        'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible', 'cloud computing', 'devops'],
        'tools': ['git', 'jira', 'agile', 'scrum', 'linux', 'bash', 'powershell', 'version control'],
        
        # FINANCE & ACCOUNTING
        'finance': ['financial analysis', 'investment', 'portfolio management', 'risk management', 'financial modeling', 'valuation', 'budgeting', 'forecasting'],
        'accounting': ['accounting', 'bookkeeping', 'tax preparation', 'auditing', 'financial reporting', 'gaap', 'quickbooks', 'excel', 'accounts payable', 'accounts receivable'],
        'banking': ['banking', 'loans', 'credit analysis', 'relationship management', 'financial products', 'compliance', 'regulatory'],
        
        # LEGAL
        'legal': ['legal research', 'contract law', 'corporate law', 'litigation', 'legal writing', 'case management', 'compliance', 'negotiation', 'legal analysis'],
        
        # HEALTHCARE
        'healthcare': ['patient care', 'nursing', 'medical procedures', 'clinical skills', 'healthcare management', 'medical terminology', 'healthcare systems', 'patient safety'],
        'medical': ['diagnosis', 'treatment', 'medical records', 'pharmacology', 'anatomy', 'physiology', 'medical equipment'],
        
        # EDUCATION
        'education': ['teaching', 'curriculum development', 'lesson planning', 'classroom management', 'educational assessment', 'student development', 'pedagogy'],
        'academic': ['research', 'academic writing', 'educational technology', 'learning management systems', 'instructional design'],
        
        # SALES & MARKETING
        'sales': ['sales management', 'lead generation', 'customer acquisition', 'negotiation', 'crm', 'sales strategy', 'business development'],
        'marketing': ['digital marketing', 'social media marketing', 'content marketing', 'seo', 'sem', 'email marketing', 'brand management', 'market research'],
        'advertising': ['advertising', 'campaign management', 'media planning', 'creative development', 'analytics', 'roi analysis'],
        
        # HUMAN RESOURCES
        'hr': ['human resources', 'talent acquisition', 'recruiting', 'employee relations', 'performance management', 'hr strategy', 'compensation', 'benefits'],
        'recruitment': ['sourcing', 'interviewing', 'candidate assessment', 'onboarding', 'talent management'],
        
        # ENGINEERING
        'mechanical': ['mechanical engineering', 'cad', 'solidworks', 'autocad', 'manufacturing', 'design', 'quality control', 'project management'],
        'civil': ['civil engineering', 'structural design', 'construction management', 'surveying', 'project planning', 'infrastructure'],
        'electrical': ['electrical engineering', 'circuit design', 'power systems', 'electronics', 'automation', 'plc programming'],
        
        # DESIGN & CREATIVE
        'design': ['graphic design', 'visual design', 'branding', 'typography', 'layout design', 'creative direction'],
        'digital_design': ['ux design', 'ui design', 'user experience', 'prototyping', 'wireframing', 'figma', 'sketch', 'adobe creative suite'],
        'creative': ['creativity', 'artistic', 'illustration', 'photography', 'video editing', 'animation'],
        
        # CULINARY & HOSPITALITY
        'culinary': ['culinary arts', 'cooking', 'food preparation', 'menu planning', 'kitchen management', 'food safety', 'recipe development'],
        'hospitality': ['customer service', 'hotel management', 'event planning', 'guest relations', 'hospitality management'],
        
        # CONSTRUCTION
        'construction': ['construction management', 'project management', 'safety management', 'scheduling', 'budgeting', 'quality control', 'site supervision'],
        'trades': ['carpentry', 'plumbing', 'electrical work', 'masonry', 'roofing', 'hvac'],
        
        # AGRICULTURE
        'agriculture': ['farming', 'crop management', 'soil science', 'agricultural technology', 'sustainability', 'livestock management', 'irrigation'],
        
        # FITNESS & WELLNESS
        'fitness': ['personal training', 'fitness coaching', 'exercise physiology', 'nutrition', 'health coaching', 'wellness programs'],
        
        # CONSULTING & BUSINESS
        'consulting': ['business consulting', 'strategy consulting', 'management consulting', 'process improvement', 'change management'],
        'business': ['business analysis', 'strategic planning', 'operations management', 'supply chain', 'logistics', 'vendor management'],
        
        # AVIATION & TRANSPORTATION
        'aviation': ['flight operations', 'aviation safety', 'aircraft maintenance', 'navigation', 'air traffic control'],
        'transportation': ['logistics', 'supply chain management', 'fleet management', 'transportation planning'],
        
        # PUBLIC RELATIONS & COMMUNICATIONS
        'communications': ['public relations', 'media relations', 'corporate communications', 'crisis management', 'press releases'],
        'writing': ['content writing', 'copywriting', 'technical writing', 'journalism', 'editing', 'proofreading'],
        
        # SOFT SKILLS
        'leadership': ['leadership', 'team management', 'mentoring', 'coaching', 'delegation', 'motivation'],
        'communication': ['communication', 'presentation', 'public speaking', 'interpersonal skills', 'collaboration'],
        'analytical': ['analytical thinking', 'problem solving', 'critical thinking', 'data analysis', 'research'],
        'organizational': ['organization', 'time management', 'multitasking', 'attention to detail', 'planning'],
        'customer_service': ['customer service', 'client relations', 'customer satisfaction', 'support', 'relationship building']
    }
    
    text_lower = text.lower()
    found_skills = []
    
    for category, skills in skill_keywords.items():
        for skill in skills:
            if skill in text_lower:
                found_skills.append(skill)
    
    return list(set(found_skills))

def extract_experience_years(text):
    """Extract years of experience from text"""
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

def extract_education(text):
    """Extract education information"""
    education_keywords = [
        'Bachelor', 'Master', 'PhD', 'B.Tech', 'M.Tech', 'MBA', 'B.S', 'M.S',
        'BSc', 'MSc', 'BE', 'ME', 'Diploma', 'Associate', 'Doctorate'
    ]
    
    education = []
    lines = text.split('\n')
    for line in lines:
        if any(keyword.lower() in line.lower() for keyword in education_keywords):
            education.append(line.strip())
    
    return education[:3] if education else []

def categorize_resume(text, skills):
    """Categorize resume based on content and skills"""
    text_lower = text.lower()
    
    # Category keywords mapping
    category_keywords = {
        'Information Technology': ['software', 'programming', 'developer', 'engineer', 'technology', 'it', 'computer', 'web', 'mobile', 'database', 'system'],
        'Accountant': ['accountant', 'accounting', 'bookkeeper', 'financial reporting', 'tax', 'audit', 'cpa', 'gaap'],
        'Finance': ['finance', 'financial analyst', 'investment', 'banking', 'portfolio', 'risk management', 'treasury'],
        'Banking': ['bank', 'banking', 'loan officer', 'credit', 'mortgage', 'financial services', 'branch manager'],
        'Advocate': ['lawyer', 'attorney', 'legal', 'law', 'litigation', 'counsel', 'paralegal', 'legal assistant'],
        'Healthcare': ['nurse', 'doctor', 'physician', 'medical', 'healthcare', 'clinical', 'hospital', 'patient care'],
        'Teacher': ['teacher', 'educator', 'professor', 'instructor', 'education', 'teaching', 'academic', 'school'],
        'Sales': ['sales', 'account manager', 'business development', 'sales representative', 'sales manager'],
        'Marketing': ['marketing', 'brand', 'advertising', 'promotion', 'campaign', 'digital marketing', 'social media'],
        'HR': ['human resources', 'hr', 'recruiter', 'talent acquisition', 'employee relations', 'hr manager'],
        'Engineering': ['engineer', 'engineering', 'mechanical', 'civil', 'electrical', 'chemical', 'industrial'],
        'Designer': ['designer', 'design', 'graphic', 'creative', 'ux', 'ui', 'visual', 'art director'],
        'Chef': ['chef', 'cook', 'culinary', 'kitchen', 'restaurant', 'food service', 'catering'],
        'Construction': ['construction', 'contractor', 'builder', 'project manager', 'site supervisor', 'foreman'],
        'Agriculture': ['agriculture', 'farming', 'agricultural', 'crop', 'livestock', 'farm manager'],
        'Fitness': ['fitness', 'trainer', 'coach', 'gym', 'exercise', 'wellness', 'health coach'],
        'Consultant': ['consultant', 'consulting', 'advisor', 'strategy', 'management consultant'],
        'Aviation': ['pilot', 'aviation', 'aircraft', 'flight', 'airline', 'air traffic'],
        'Business Development': ['business development', 'partnership', 'growth', 'strategic partnerships'],
        'Public Relations': ['public relations', 'pr', 'communications', 'media relations', 'press'],
        'Digital Media': ['digital media', 'content creator', 'social media manager', 'digital marketing'],
        'BPO': ['bpo', 'call center', 'customer service', 'support', 'outsourcing'],
        'Automobile': ['automotive', 'automobile', 'car', 'vehicle', 'auto mechanic', 'automotive engineer'],
        'Apparel': ['fashion', 'apparel', 'clothing', 'textile', 'garment', 'fashion designer'],
        'Arts': ['artist', 'arts', 'creative', 'painter', 'sculptor', 'art', 'fine arts']
    }
    
    # Score each category
    category_scores = {}
    for category, keywords in category_keywords.items():
        score = 0
        for keyword in keywords:
            if keyword in text_lower:
                score += text_lower.count(keyword)
        
        # Boost score if skills match category
        skill_boost = 0
        for skill in skills:
            if any(keyword in skill.lower() for keyword in keywords):
                skill_boost += 1
        
        category_scores[category] = score + (skill_boost * 2)
    
    # Return the category with highest score, or 'General' if no clear match
    if category_scores:
        best_category = max(category_scores, key=category_scores.get)
        if category_scores[best_category] > 0:
            return best_category
    
    return 'General'

def calculate_resume_score(resume_skills, resume_exp, job_skills, job_exp, resume_text, job_text, model):
    """Calculate comprehensive resume score"""
    # Skill match score
    resume_skills_set = set([s.lower() for s in resume_skills])
    job_skills_set = set([s.lower() for s in job_skills])
    
    if job_skills_set:
        skill_match = len(resume_skills_set.intersection(job_skills_set)) / len(job_skills_set) * 100
    else:
        skill_match = 0
    
    # Experience match score
    if resume_exp and job_exp:
        exp_match = min(resume_exp / job_exp * 100, 100)
    else:
        exp_match = 50
    
    # Semantic similarity
    resume_emb = model.encode(resume_text[:5000])
    job_emb = model.encode(job_text[:5000])
    semantic_sim = cosine_similarity(resume_emb.reshape(1, -1), job_emb.reshape(1, -1))[0][0] * 100
    
    # Overall score (weighted)
    overall_score = semantic_sim * 0.4 + skill_match * 0.4 + exp_match * 0.2
    
    return {
        'overall_score': round(overall_score, 2),
        'skill_match': round(skill_match, 2),
        'experience_match': round(exp_match, 2),
        'semantic_similarity': round(semantic_sim, 2)
    }

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/resume.png", width=100)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", 
    ["🏠 Home", "📤 Upload Resume", "👤 Resume Matcher", "💼 Job Searcher", "📊 Analytics", "ℹ️ About"])

# Comprehensive job database covering all industries
job_database = pd.DataFrame([
    # TECHNOLOGY
    {
        'job_id': 'JOB001',
        'title': 'Senior Python Developer',
        'company': 'Tech Corp',
        'description': 'Looking for an experienced Python developer with expertise in Django, Flask, REST APIs, and cloud technologies.',
        'required_skills': ['python', 'django', 'flask', 'aws', 'docker', 'sql', 'programming'],
        'experience_required': 5,
        'category': 'Information Technology',
        'salary_range': '$100k-$140k',
        'location': 'San Francisco, CA'
    },
    {
        'job_id': 'JOB002',
        'title': 'Machine Learning Engineer',
        'company': 'AI Innovations',
        'description': 'Seeking ML engineer with strong background in deep learning, NLP, and computer vision.',
        'required_skills': ['python', 'machine learning', 'tensorflow', 'pytorch', 'nlp', 'deep learning', 'data science'],
        'experience_required': 3,
        'category': 'Information Technology',
        'salary_range': '$120k-$160k',
        'location': 'Remote'
    },
    # FINANCE & ACCOUNTING
    {
        'job_id': 'JOB003',
        'title': 'Senior Accountant',
        'company': 'Financial Services LLC',
        'description': 'Experienced accountant needed for financial reporting, tax preparation, and audit support.',
        'required_skills': ['accounting', 'financial reporting', 'tax preparation', 'excel', 'quickbooks', 'gaap', 'auditing'],
        'experience_required': 4,
        'category': 'Accountant',
        'salary_range': '$65k-$85k',
        'location': 'Chicago, IL'
    },
    {
        'job_id': 'JOB004',
        'title': 'Financial Analyst',
        'company': 'Investment Group',
        'description': 'Analyze financial data, prepare investment reports, and support strategic decision making.',
        'required_skills': ['financial analysis', 'excel', 'modeling', 'reporting', 'investment', 'valuation'],
        'experience_required': 3,
        'category': 'Finance',
        'salary_range': '$70k-$95k',
        'location': 'New York, NY'
    },
    # LEGAL
    {
        'job_id': 'JOB005',
        'title': 'Corporate Lawyer',
        'company': 'Legal Associates',
        'description': 'Handle corporate legal matters, contract negotiations, and regulatory compliance.',
        'required_skills': ['legal research', 'contract law', 'corporate law', 'litigation', 'negotiation', 'compliance'],
        'experience_required': 5,
        'category': 'Advocate',
        'salary_range': '$120k-$180k',
        'location': 'Los Angeles, CA'
    },
    # HEALTHCARE
    {
        'job_id': 'JOB006',
        'title': 'Registered Nurse',
        'company': 'City Medical Center',
        'description': 'Provide patient care, administer medications, and collaborate with healthcare team.',
        'required_skills': ['patient care', 'nursing', 'medical procedures', 'healthcare', 'clinical skills', 'communication'],
        'experience_required': 2,
        'category': 'Healthcare',
        'salary_range': '$65k-$85k',
        'location': 'Houston, TX'
    },
    # EDUCATION
    {
        'job_id': 'JOB007',
        'title': 'High School Mathematics Teacher',
        'company': 'Lincoln High School',
        'description': 'Teach mathematics, develop lesson plans, and assess student progress.',
        'required_skills': ['teaching', 'mathematics', 'curriculum development', 'classroom management', 'education'],
        'experience_required': 2,
        'category': 'Teacher',
        'salary_range': '$45k-$65k',
        'location': 'Denver, CO'
    },
    # SALES & MARKETING
    {
        'job_id': 'JOB008',
        'title': 'Sales Manager',
        'company': 'Enterprise Solutions',
        'description': 'Lead sales team, develop strategies, and drive revenue growth.',
        'required_skills': ['sales management', 'leadership', 'business development', 'negotiation', 'crm'],
        'experience_required': 4,
        'category': 'Sales',
        'salary_range': '$70k-$100k',
        'location': 'Atlanta, GA'
    },
    {
        'job_id': 'JOB009',
        'title': 'Digital Marketing Specialist',
        'company': 'Creative Agency',
        'description': 'Develop digital marketing campaigns, manage social media, and analyze performance.',
        'required_skills': ['digital marketing', 'social media', 'seo', 'content marketing', 'analytics', 'advertising'],
        'experience_required': 2,
        'category': 'Digital Media',
        'salary_range': '$50k-$70k',
        'location': 'Miami, FL'
    },
    # HUMAN RESOURCES
    {
        'job_id': 'JOB010',
        'title': 'HR Business Partner',
        'company': 'Global Corporation',
        'description': 'Partner with business leaders on HR strategy, talent management, and organizational development.',
        'required_skills': ['human resources', 'talent management', 'employee relations', 'hr strategy', 'leadership'],
        'experience_required': 5,
        'category': 'HR',
        'salary_range': '$80k-$110k',
        'location': 'Seattle, WA'
    },
    # ENGINEERING
    {
        'job_id': 'JOB011',
        'title': 'Mechanical Engineer',
        'company': 'Manufacturing Corp',
        'description': 'Design mechanical systems, oversee manufacturing processes, and ensure quality standards.',
        'required_skills': ['mechanical engineering', 'cad', 'manufacturing', 'design', 'project management', 'quality control'],
        'experience_required': 4,
        'category': 'Engineering',
        'salary_range': '$75k-$105k',
        'location': 'Detroit, MI'
    },
    # DESIGN
    {
        'job_id': 'JOB012',
        'title': 'UX/UI Designer',
        'company': 'Tech Startup',
        'description': 'Design user interfaces and experiences for web and mobile applications.',
        'required_skills': ['ux design', 'ui design', 'prototyping', 'user research', 'figma', 'design thinking'],
        'experience_required': 3,
        'category': 'Designer',
        'salary_range': '$70k-$95k',
        'location': 'New York, NY'
    },
    # CULINARY
    {
        'job_id': 'JOB013',
        'title': 'Executive Chef',
        'company': 'Fine Dining Restaurant',
        'description': 'Lead kitchen operations, menu development, and culinary team management.',
        'required_skills': ['culinary arts', 'menu planning', 'kitchen management', 'food safety', 'leadership', 'creativity'],
        'experience_required': 6,
        'category': 'Chef',
        'salary_range': '$60k-$85k',
        'location': 'Las Vegas, NV'
    },
    # CONSTRUCTION
    {
        'job_id': 'JOB014',
        'title': 'Construction Project Manager',
        'company': 'BuildRight Construction',
        'description': 'Manage construction projects from planning to completion, coordinate teams and resources.',
        'required_skills': ['project management', 'construction', 'scheduling', 'budgeting', 'leadership', 'safety'],
        'experience_required': 5,
        'category': 'Construction',
        'salary_range': '$75k-$105k',
        'location': 'Dallas, TX'
    },
    # AGRICULTURE
    {
        'job_id': 'JOB015',
        'title': 'Agricultural Specialist',
        'company': 'AgriTech Solutions',
        'description': 'Provide expertise in crop management, soil analysis, and sustainable farming practices.',
        'required_skills': ['agriculture', 'crop management', 'soil science', 'farming', 'sustainability', 'research'],
        'experience_required': 3,
        'category': 'Agriculture',
        'salary_range': '$50k-$70k',
        'location': 'Iowa City, IA'
    }
])

# Helper functions
def calculate_match_score(resume_skills, job_skills, resume_exp, job_exp, resume_emb, job_emb):
    # Parse skills
    if isinstance(resume_skills, str):
        resume_skills = eval(resume_skills) if resume_skills.startswith('[') else []
    
    resume_skills_set = set([s.lower() for s in resume_skills])
    job_skills_set = set([s.lower() for s in job_skills])
    
    # Skill match
    skill_match = len(resume_skills_set.intersection(job_skills_set)) / len(job_skills_set) * 100 if job_skills_set else 0
    
    # Experience match
    exp_match = min(resume_exp / job_exp * 100, 100) if resume_exp and job_exp else 50
    
    # Semantic similarity
    semantic = cosine_similarity(resume_emb.reshape(1, -1), job_emb.reshape(1, -1))[0][0] * 100
    
    # Weighted score
    final_score = semantic * 0.4 + skill_match * 0.4 + exp_match * 0.2
    
    return {
        'final_score': round(final_score, 2),
        'skill_match': round(skill_match, 2),
        'exp_match': round(exp_match, 2),
        'semantic': round(semantic, 2)
    }

# HOME PAGE
if page == "🏠 Home":
    st.markdown('<p class="main-header">🎯 AI-Powered Resume Matcher</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the Next-Gen Resume Matching Platform!
    
    This intelligent system uses **AI and Machine Learning** to match resumes with job opportunities.
    
    #### 🌟 Key Features:
    - **Smart Resume Parsing**: Extract skills, experience, education, and more
    - **AI-Powered Matching**: Uses semantic embeddings for accurate job recommendations
    - **Multi-Factor Scoring**: Considers skills, experience, and context
    - **Interactive Visualizations**: Explore insights with beautiful charts
    - **Real-time Recommendations**: Get instant job matches
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Resumes", len(resume_df), "📄")
    
    with col2:
        st.metric("Job Categories", resume_df['category'].nunique(), "📂")
    
    with col3:
        total_skills = 0
        for skills in resume_df['skills']:
            try:
                skill_list = eval(skills) if isinstance(skills, str) else skills
                total_skills += len(skill_list)
            except:
                pass
        st.metric("Avg Skills", f"{total_skills/len(resume_df):.1f}", "⚡")
    
    with col4:
        avg_exp = resume_df['experience_years'].mean()
        st.metric("Avg Experience", f"{avg_exp:.1f} yrs", "📈")
    
    st.markdown("---")
    st.info("👈 Use the sidebar to navigate through different features!")

# UPLOAD RESUME PAGE
elif page == "📤 Upload Resume":
    st.markdown('<p class="main-header">📤 Upload Your Resume</p>', unsafe_allow_html=True)
    st.markdown("### Get personalized job recommendations and resume analysis")
    
    # File upload
    uploaded_file = st.file_uploader("Choose your resume (PDF)", type="pdf")
    
    if uploaded_file is not None:
        # Extract text from uploaded resume
        resume_text = extract_text_from_pdf(uploaded_file)
        
        if resume_text:
            # Parse resume
            resume_skills = extract_skills_from_text(resume_text)
            resume_exp = extract_experience_years(resume_text)
            resume_education = extract_education(resume_text)
            resume_category = categorize_resume(resume_text, resume_skills)
            
            # Display parsed information
            st.markdown("### 📋 Parsed Resume Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Extracted Skills")
                if resume_skills:
                    st.write(f"**Found {len(resume_skills)} skills:**")
                    skills_text = ", ".join(resume_skills)
                    st.write(skills_text)
                else:
                    st.write("No specific skills detected")
                
                st.markdown("#### 📂 Detected Category")
                st.write(f"**{resume_category}**")
                
                st.markdown("#### 📚 Education")
                if resume_education:
                    for edu in resume_education:
                        st.write(f"• {edu}")
                else:
                    st.write("No education information detected")
            
            with col2:
                st.markdown("#### 💼 Experience")
                if resume_exp:
                    st.write(f"**{resume_exp} years of experience**")
                else:
                    st.write("Experience not clearly specified")
                
                st.markdown("#### 📊 Resume Stats")
                word_count = len(resume_text.split())
                st.write(f"• Word count: {word_count}")
                st.write(f"• Character count: {len(resume_text)}")
            
            st.markdown("---")
            
            # Job recommendations
            st.markdown("### 💼 Personalized Job Recommendations")
            
            # Calculate matches with job database
            recommendations = []
            for idx, job in job_database.iterrows():
                job_text = f"{job['title']} {job['description']} {' '.join(job['required_skills'])}"
                
                scores = calculate_resume_score(
                    resume_skills,
                    resume_exp,
                    job['required_skills'],
                    job['experience_required'],
                    resume_text,
                    job_text,
                    model
                )
                
                recommendations.append({
                    **job.to_dict(),
                    **scores
                })
            
            # Sort by overall score
            recommendations = sorted(recommendations, key=lambda x: x['overall_score'], reverse=True)
            
            # Display top recommendations
            for i, rec in enumerate(recommendations[:5], 1):
                with st.container():
                    st.markdown(f"""
                    <div class="job-card">
                        <h3>{i}. {rec['title']} - {rec['company']}</h3>
                        <p><strong>📍 Location:</strong> {rec['location']} | <strong>💰 Salary:</strong> {rec['salary_range']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Score metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Overall Score", f"{rec['overall_score']}%", "🎯")
                    col2.metric("Skill Match", f"{rec['skill_match']}%", "⚡")
                    col3.metric("Experience Match", f"{rec['experience_match']}%", "📈")
                    col4.metric("Semantic Match", f"{rec['semantic_similarity']}%", "🧠")
                    
                    # Detailed analysis
                    with st.expander("📊 Detailed Analysis"):
                        st.write(f"**Job Description:** {rec['description']}")
                        st.write(f"**Required Skills:** {', '.join(rec['required_skills'])}")
                        st.write(f"**Experience Required:** {rec['experience_required']} years")
                        
                        # Skills comparison
                        user_skills_set = set([s.lower() for s in resume_skills])
                        job_skills_set = set([s.lower() for s in rec['required_skills']])
                        
                        matching_skills = user_skills_set.intersection(job_skills_set)
                        missing_skills = job_skills_set - user_skills_set
                        
                        if matching_skills:
                            st.success(f"**Matching Skills:** {', '.join(matching_skills)}")
                        if missing_skills:
                            st.warning(f"**Skills to Develop:** {', '.join(missing_skills)}")
                    
                    st.markdown("---")
            
            # Resume improvement suggestions
            st.markdown("### 🚀 Resume Improvement Suggestions")
            
            # Analyze all job requirements to suggest skills
            all_job_skills = []
            for _, job in job_database.iterrows():
                all_job_skills.extend(job['required_skills'])
            
            skill_frequency = Counter(all_job_skills)
            top_market_skills = [skill for skill, _ in skill_frequency.most_common(10)]
            
            user_skills_lower = [s.lower() for s in resume_skills]
            suggested_skills = [skill for skill in top_market_skills if skill.lower() not in user_skills_lower]
            
            if suggested_skills:
                st.info(f"**Consider adding these in-demand skills:** {', '.join(suggested_skills[:5])}")
            
            # Experience suggestions
            avg_exp_required = job_database['experience_required'].mean()
            if resume_exp and resume_exp < avg_exp_required:
                st.info(f"**Experience Insight:** Average job requirement is {avg_exp_required:.1f} years. Consider highlighting relevant projects or internships.")
            
        else:
            st.error("Could not extract text from the PDF. Please ensure it's a valid PDF file.")

# RESUME MATCHER PAGE
elif page == "👤 Resume Matcher":
    st.markdown('<p class="main-header">👤 Resume Job Matcher</p>', unsafe_allow_html=True)
    st.markdown("### Find the best job matches for a resume")
    
    # Resume selection
    resume_ids = resume_df['resume_id'].tolist()
    selected_resume = st.selectbox("Select a Resume:", resume_ids)
    
    if selected_resume:
        resume_idx = resume_df[resume_df['resume_id'] == selected_resume].index[0]
        resume = resume_df.iloc[resume_idx]
        resume_embedding = embeddings[resume_idx]
        
        # Display resume info
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### 📋 Resume Details")
            st.write(f"**Category:** {resume['category']}")
            st.write(f"**Experience:** {resume['experience_years']} years" if pd.notna(resume['experience_years']) else "**Experience:** Not specified")
            
            if pd.notna(resume['skills']):
                try:
                    skills = eval(resume['skills']) if isinstance(resume['skills'], str) else resume['skills']
                    st.write(f"**Skills ({len(skills)}):**")
                    st.write(", ".join(skills[:10]))
                    if len(skills) > 10:
                        with st.expander("View all skills"):
                            st.write(", ".join(skills))
                except:
                    pass
        
        with col2:
            st.markdown("#### 📝 Resume Preview")
            if pd.notna(resume['resume_text']):
                st.text_area("", resume['resume_text'][:500] + "...", height=200, disabled=True)
        
        st.markdown("---")
        st.markdown("### 💼 Top Job Recommendations")
        
        # Calculate matches for all jobs
        recommendations = []
        for idx, job in job_database.iterrows():
            job_text = f"{job['title']} {job['description']} {' '.join(job['required_skills'])}"
            job_emb = model.encode(job_text)
            
            scores = calculate_match_score(
                resume['skills'],
                job['required_skills'],
                resume['experience_years'],
                job['experience_required'],
                resume_embedding,
                job_emb
            )
            
            recommendations.append({
                **job.to_dict(),
                **scores
            })
        
        # Sort by score
        recommendations = sorted(recommendations, key=lambda x: x['final_score'], reverse=True)
        
        # Display top 5
        for i, rec in enumerate(recommendations[:5], 1):
            with st.container():
                st.markdown(f"""
                <div class="job-card">
                    <h3>{i}. {rec['title']} - {rec['company']}</h3>
                    <p><strong>📍 Location:</strong> {rec['location']} | <strong>💰 Salary:</strong> {rec['salary_range']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Match Score", f"{rec['final_score']}%", "🎯")
                col2.metric("Skill Match", f"{rec['skill_match']}%", "⚡")
                col3.metric("Experience Match", f"{rec['exp_match']}%", "📈")
                col4.metric("Semantic Match", f"{rec['semantic']}%", "🧠")
                
                with st.expander("View Details"):
                    st.write(f"**Description:** {rec['description']}")
                    st.write(f"**Required Skills:** {', '.join(rec['required_skills'])}")
                    st.write(f"**Experience Required:** {rec['experience_required']} years")
                
                st.markdown("---")

# JOB SEARCHER PAGE
elif page == "💼 Job Searcher":
    st.markdown('<p class="main-header">💼 Candidate Finder</p>', unsafe_allow_html=True)
    st.markdown("### Find the best candidates for a job")
    
    # Job selection
    job_titles = job_database['title'].tolist()
    selected_job = st.selectbox("Select a Job:", job_titles)
    
    if selected_job:
        job = job_database[job_database['title'] == selected_job].iloc[0]
        
        # Display job info
        st.markdown(f"### {job['title']}")
        col1, col2, col3 = st.columns(3)
        col1.write(f"**Company:** {job['company']}")
        col2.write(f"**Location:** {job['location']}")
        col3.write(f"**Salary:** {job['salary_range']}")
        
        st.write(f"**Description:** {job['description']}")
        st.write(f"**Required Skills:** {', '.join(job['required_skills'])}")
        st.write(f"**Experience Required:** {job['experience_required']} years")
        
        st.markdown("---")
        st.markdown("### 👥 Top Candidates")
        
        # Generate job embedding
        job_text = f"{job['title']} {job['description']} {' '.join(job['required_skills'])}"
        job_emb = model.encode(job_text)
        
        # Calculate matches
        candidates = []
        for idx, resume in resume_df.iterrows():
            scores = calculate_match_score(
                resume['skills'],
                job['required_skills'],
                resume['experience_years'],
                job['experience_required'],
                embeddings[idx],
                job_emb
            )
            
            candidates.append({
                'resume_id': resume['resume_id'],
                'category': resume['category'],
                'skills': resume['skills'],
                'experience': resume['experience_years'],
                **scores
            })
        
        # Sort and display top 10
        candidates = sorted(candidates, key=lambda x: x['final_score'], reverse=True)
        
        for i, candidate in enumerate(candidates[:10], 1):
            col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 1, 2])
            
            col1.write(f"**#{i}**")
            col2.write(f"**{candidate['resume_id']}**")
            col3.metric("Match", f"{candidate['final_score']}%")
            col4.write(f"{candidate['experience']} yrs" if pd.notna(candidate['experience']) else "N/A")
            
            try:
                skills = eval(candidate['skills']) if isinstance(candidate['skills'], str) else candidate['skills']
                col5.write(f"{len(skills)} skills")
            except:
                col5.write("N/A")
            
            st.markdown("---")

# ANALYTICS PAGE
elif page == "📊 Analytics":
    st.markdown('<p class="main-header">📊 Resume Analytics</p>', unsafe_allow_html=True)
    
    # Category distribution
    st.markdown("### 📂 Category Distribution")
    category_counts = resume_df['category'].value_counts()
    
    fig = px.bar(
        x=category_counts.index,
        y=category_counts.values,
        labels={'x': 'Category', 'y': 'Count'},
        title='Resume Distribution by Category',
        color=category_counts.values,
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Experience distribution
    st.markdown("### 📈 Experience Distribution")
    exp_data = resume_df['experience_years'].dropna()
    
    fig = px.histogram(
        exp_data,
        nbins=20,
        title='Distribution of Years of Experience',
        labels={'value': 'Years of Experience', 'count': 'Number of Resumes'},
        color_discrete_sequence=['#1E88E5']
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Skills analysis
    st.markdown("### ⚡ Top Skills")
    all_skills = []
    for skills in resume_df['skills']:
        if pd.notna(skills):
            try:
                skill_list = eval(skills) if isinstance(skills, str) else skills
                all_skills.extend(skill_list)
            except:
                pass
    
    from collections import Counter
    skill_counts = Counter(all_skills)
    top_20_skills = dict(skill_counts.most_common(20))
    
    fig = px.bar(
        x=list(top_20_skills.values()),
        y=list(top_20_skills.keys()),
        orientation='h',
        title='Top 20 Skills in Dataset',
        labels={'x': 'Frequency', 'y': 'Skill'},
        color=list(top_20_skills.values()),
        color_continuous_scale='Turbo'
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

# ABOUT PAGE
elif page == "ℹ️ About":
    st.markdown('<p class="main-header">ℹ️ About This Project</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Project Overview
    
    This **AI-Powered Resume Matcher** is a comprehensive system that uses advanced NLP and machine learning 
    techniques to intelligently match resumes with job opportunities.
    
    ### 🔧 Technologies Used
    
    - **Python**: Core programming language
    - **Sentence Transformers**: For semantic text embeddings
    - **spaCy**: Natural language processing and entity extraction
    - **Scikit-learn**: Machine learning algorithms
    - **Streamlit**: Interactive web interface
    - **Plotly**: Interactive visualizations
    - **Pandas & NumPy**: Data manipulation and analysis
    
    ### 📊 How It Works
    
    1. **Resume Parsing**: Extracts key information from PDFs (skills, experience, education, etc.)
    2. **Feature Extraction**: Uses NLP to identify entities, skills, and qualifications
    3. **Embedding Generation**: Creates semantic vector representations of resumes and jobs
    4. **Similarity Matching**: Calculates multi-factor match scores:
       - Semantic similarity (40%)
       - Skill overlap (40%)
       - Experience match (20%)
    5. **Ranking & Recommendations**: Returns top matches with detailed breakdowns
    
    ### 🚀 Unique Features
    
    - ✅ Multi-modal text extraction from complex PDFs
    - ✅ Advanced NLP for entity and skill recognition
    - ✅ Semantic understanding beyond keyword matching
    - ✅ Interactive visualizations and analytics
    - ✅ Bidirectional matching (resume→jobs and job→candidates)
    - ✅ Real-time processing and recommendations
    
    ### 👨‍💻 Developer
    
    Created with ❤️ for intelligent recruitment automation
    """)
    
    st.success("🌟 Star this project if you found it useful!")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** This system uses AI to understand context, not just keywords!")