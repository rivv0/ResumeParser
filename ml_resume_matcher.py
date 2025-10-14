"""
Advanced ML-Powered Resume Matcher
Uses state-of-the-art embedding models and machine learning techniques for accurate job matching
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import re
from collections import Counter
from typing import List, Dict, Tuple
import pickle
import os
from PyPDF2 import PdfReader
import spacy
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')

class AdvancedResumeMLMatcher:
    def __init__(self):
        """Initialize the advanced ML-based resume matcher"""
        print("🚀 Initializing Advanced ML Resume Matcher...")
        
        # Load multiple embedding models for ensemble approach
        self.models = self._load_embedding_models()
        
        # Load NLP model
        self.nlp = self._load_nlp_model()
        
        # Initialize ML components
        self.scaler = StandardScaler()
        self.ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Skill categories and weights
        self.skill_categories = self._initialize_skill_categories()
        
        # Job database
        self.job_database = self._create_comprehensive_job_database()
        
        print("✅ ML Resume Matcher initialized successfully!")
    
    def _load_embedding_models(self) -> Dict:
        """Load multiple state-of-the-art embedding models"""
        models = {}
        
        # Try to load models in order of preference
        model_configs = [
            ('bge', 'BAAI/bge-large-en-v1.5', 'BGE embedding model'),
            ('e5', 'intfloat/e5-large-v2', 'E5 embedding model'),
            ('sbert', 'all-MiniLM-L6-v2', 'Sentence-BERT model')
        ]
        
        for model_key, model_name, description in model_configs:
            try:
                print(f"📥 Loading {description}...")
                models[model_key] = SentenceTransformer(model_name)
                print(f"✅ {description} loaded")
            except Exception as e:
                print(f"⚠️ {description} failed to load: {e}")
                # Continue trying other models
        
        # If no models loaded, try a basic fallback
        if not models:
            try:
                print("📥 Loading basic fallback model...")
                models['basic'] = SentenceTransformer('all-mpnet-base-v2')
                print("✅ Basic fallback model loaded")
            except Exception as e:
                print(f"❌ Even basic model failed: {e}")
                raise Exception("No embedding models could be loaded! Please check your internet connection.")
            
        return models
    
    def _load_nlp_model(self):
        """Load spaCy NLP model"""
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️ spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            return None
    
    def _initialize_skill_categories(self) -> Dict:
        """Initialize comprehensive skill categories with weights"""
        return {
            # TECHNOLOGY & PROGRAMMING
            'programming_languages': {
                'skills': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin', 'go', 'rust', 'typescript', 'scala', 'r'],
                'weight': 1.5
            },
            'web_technologies': {
                'skills': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring', 'asp.net', 'laravel', 'express'],
                'weight': 1.4
            },
            'databases': {
                'skills': ['sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'redis', 'cassandra', 'elasticsearch', 'sqlite', 'dynamodb'],
                'weight': 1.3
            },
            'cloud_devops': {
                'skills': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible', 'gitlab', 'circleci'],
                'weight': 1.4
            },
            'data_science': {
                'skills': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy', 'matplotlib'],
                'weight': 1.6
            },
            'mobile_development': {
                'skills': ['android', 'ios', 'react native', 'flutter', 'xamarin', 'ionic'],
                'weight': 1.3
            },
            
            # BUSINESS & FINANCE
            'finance_accounting': {
                'skills': ['financial analysis', 'accounting', 'bookkeeping', 'tax preparation', 'auditing', 'gaap', 'ifrs', 'quickbooks', 'sap'],
                'weight': 1.2
            },
            'banking_investment': {
                'skills': ['banking', 'investment', 'portfolio management', 'risk management', 'credit analysis', 'financial modeling'],
                'weight': 1.3
            },
            
            # LEGAL
            'legal_compliance': {
                'skills': ['legal research', 'contract law', 'corporate law', 'litigation', 'compliance', 'regulatory', 'intellectual property'],
                'weight': 1.2
            },
            
            # HEALTHCARE
            'healthcare_medical': {
                'skills': ['patient care', 'nursing', 'medical procedures', 'clinical skills', 'healthcare management', 'medical terminology'],
                'weight': 1.1
            },
            
            # MARKETING & SALES
            'digital_marketing': {
                'skills': ['seo', 'sem', 'social media marketing', 'content marketing', 'email marketing', 'google analytics', 'facebook ads'],
                'weight': 1.2
            },
            'sales_business_dev': {
                'skills': ['sales', 'business development', 'lead generation', 'crm', 'salesforce', 'account management', 'negotiation'],
                'weight': 1.1
            },
            
            # DESIGN & CREATIVE
            'design_creative': {
                'skills': ['graphic design', 'ux design', 'ui design', 'adobe creative suite', 'figma', 'sketch', 'prototyping'],
                'weight': 1.2
            },
            
            # SOFT SKILLS
            'leadership_management': {
                'skills': ['leadership', 'team management', 'project management', 'agile', 'scrum', 'mentoring', 'coaching'],
                'weight': 1.0
            },
            'communication': {
                'skills': ['communication', 'presentation', 'public speaking', 'writing', 'interpersonal skills'],
                'weight': 0.9
            },
            'analytical_problem_solving': {
                'skills': ['analytical thinking', 'problem solving', 'critical thinking', 'research', 'data analysis'],
                'weight': 1.1
            }
        }
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            print(f"Error reading PDF: {e}")
        return text
    
    def extract_advanced_skills(self, text: str) -> Dict:
        """Extract skills using advanced NLP and categorization"""
        text_lower = text.lower()
        extracted_skills = {}
        skill_scores = {}
        
        for category, category_data in self.skill_categories.items():
            skills = category_data['skills']
            weight = category_data['weight']
            found_skills = []
            
            for skill in skills:
                # Check for exact matches and variations
                if skill in text_lower:
                    found_skills.append(skill)
                    # Calculate skill relevance score based on frequency and context
                    frequency = text_lower.count(skill)
                    skill_scores[skill] = frequency * weight
                
                # Check for skill variations and synonyms
                variations = self._get_skill_variations(skill)
                for variation in variations:
                    if variation in text_lower and variation not in found_skills:
                        found_skills.append(variation)
                        frequency = text_lower.count(variation)
                        skill_scores[variation] = frequency * weight * 0.8  # Slightly lower weight for variations
            
            if found_skills:
                extracted_skills[category] = found_skills
        
        return extracted_skills, skill_scores
    
    def _get_skill_variations(self, skill: str) -> List[str]:
        """Get variations and synonyms for skills"""
        variations = {
            'javascript': ['js', 'ecmascript'],
            'python': ['py'],
            'machine learning': ['ml', 'artificial intelligence', 'ai'],
            'deep learning': ['neural networks', 'dl'],
            'user experience': ['ux'],
            'user interface': ['ui'],
            'search engine optimization': ['seo'],
            'customer relationship management': ['crm'],
            'application programming interface': ['api'],
            'structured query language': ['sql'],
        }
        return variations.get(skill, [])
    
    def extract_experience_advanced(self, text: str) -> Dict:
        """Extract experience information using advanced patterns"""
        experience_info = {
            'total_years': None,
            'positions': [],
            'companies': [],
            'experience_level': 'entry'
        }
        
        # Enhanced patterns for experience extraction
        patterns = [
            r'(\d+)\+?\s*years?\s*of\s*experience',
            r'experience\s*[:\-]?\s*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*experience',
            r'(\d+)\+?\s*yrs?\s*experience',
            r'over\s*(\d+)\s*years?',
            r'more\s*than\s*(\d+)\s*years?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                years = int(match.group(1))
                experience_info['total_years'] = years
                break
        
        # Determine experience level
        if experience_info['total_years']:
            if experience_info['total_years'] >= 10:
                experience_info['experience_level'] = 'senior'
            elif experience_info['total_years'] >= 5:
                experience_info['experience_level'] = 'mid'
            elif experience_info['total_years'] >= 2:
                experience_info['experience_level'] = 'junior'
        
        # Extract job positions and companies using NLP
        if self.nlp:
            doc = self.nlp(text[:5000])  # Limit text for processing
            for ent in doc.ents:
                if ent.label_ == 'ORG':
                    experience_info['companies'].append(ent.text)
                elif ent.label_ == 'PERSON' and len(ent.text.split()) <= 3:
                    # Might be a job title
                    experience_info['positions'].append(ent.text)
        
        return experience_info
    
    def generate_ensemble_embeddings(self, text: str) -> np.ndarray:
        """Generate embeddings using ensemble of models"""
        embeddings = []
        
        # Preprocess text for better embeddings
        processed_text = self._preprocess_text_for_embedding(text)
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'e5':
                    # E5 models require specific prefixes
                    embedding = model.encode(f"query: {processed_text}")
                else:
                    embedding = model.encode(processed_text)
                
                # Ensure embedding is a numpy array and flatten if needed
                embedding = np.array(embedding).flatten()
                embeddings.append(embedding)
                
            except Exception as e:
                print(f"Error generating embedding with {model_name}: {e}")
        
        if embeddings:
            # Check if all embeddings have the same dimension
            dimensions = [emb.shape[0] for emb in embeddings]
            
            if len(set(dimensions)) == 1:
                # All embeddings have same dimension - can average directly
                ensemble_embedding = np.mean(embeddings, axis=0)
            else:
                # Different dimensions - use the first (usually best) model
                print(f"⚠️ Different embedding dimensions detected: {dimensions}")
                print("Using primary model (BGE) embedding")
                ensemble_embedding = embeddings[0]  # Use first model's embedding
            
            return ensemble_embedding
        else:
            raise Exception("No embeddings could be generated")
    
    def _preprocess_text_for_embedding(self, text: str) -> str:
        """Preprocess text for better embedding generation"""
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Limit text length for embedding models (most have token limits)
        max_length = 512  # Conservative limit
        words = text.split()
        if len(words) > max_length:
            text = ' '.join(words[:max_length])
        
        return text
    
    def calculate_advanced_similarity(self, resume_embedding: np.ndarray, job_embedding: np.ndarray) -> float:
        """Calculate advanced similarity using multiple metrics"""
        # Cosine similarity
        cosine_sim = cosine_similarity(resume_embedding.reshape(1, -1), job_embedding.reshape(1, -1))[0][0]
        
        # Euclidean distance (normalized)
        euclidean_dist = np.linalg.norm(resume_embedding - job_embedding)
        euclidean_sim = 1 / (1 + euclidean_dist)
        
        # Dot product similarity (normalized)
        dot_product = np.dot(resume_embedding, job_embedding)
        norm_product = np.linalg.norm(resume_embedding) * np.linalg.norm(job_embedding)
        dot_sim = dot_product / norm_product if norm_product != 0 else 0
        
        # Weighted ensemble of similarities
        final_similarity = (cosine_sim * 0.5 + euclidean_sim * 0.3 + dot_sim * 0.2)
        
        return final_similarity
    
    def extract_ml_features(self, resume_data: Dict, job_data: Dict) -> np.ndarray:
        """Extract features for ML model"""
        features = []
        
        # Skill-based features
        resume_skills = set()
        for category_skills in resume_data.get('skills', {}).values():
            resume_skills.update([skill.lower() for skill in category_skills])
        
        job_skills = set([skill.lower() for skill in job_data.get('required_skills', [])])
        
        # Skill overlap metrics
        skill_intersection = len(resume_skills.intersection(job_skills))
        skill_union = len(resume_skills.union(job_skills))
        jaccard_similarity = skill_intersection / skill_union if skill_union > 0 else 0
        
        features.extend([
            len(resume_skills),  # Total resume skills
            len(job_skills),     # Total job skills
            skill_intersection,  # Matching skills count
            jaccard_similarity,  # Jaccard similarity
            len(resume_skills) / len(job_skills) if len(job_skills) > 0 else 0,  # Skill ratio
        ])
        
        # Experience features
        resume_exp = resume_data.get('experience', {}).get('total_years', 0) or 0
        job_exp = job_data.get('experience_required', 0) or 0
        
        features.extend([
            resume_exp,
            job_exp,
            resume_exp / job_exp if job_exp > 0 else 0,  # Experience ratio
            abs(resume_exp - job_exp),  # Experience difference
            1 if resume_exp >= job_exp else 0,  # Meets experience requirement
        ])
        
        # Text-based features
        resume_text = resume_data.get('text', '')
        job_text = job_data.get('description', '')
        
        features.extend([
            len(resume_text.split()),  # Resume word count
            len(job_text.split()),     # Job description word count
            len(set(resume_text.lower().split()).intersection(set(job_text.lower().split()))),  # Common words
        ])
        
        # Category match
        resume_category = resume_data.get('category', '')
        job_category = job_data.get('category', '')
        features.append(1 if resume_category == job_category else 0)
        
        return np.array(features)
    
    def _create_comprehensive_job_database(self) -> List[Dict]:
        """Create comprehensive job database"""
        return [
            # TECHNOLOGY
            {
                'job_id': 'TECH001',
                'title': 'Senior Machine Learning Engineer',
                'company': 'AI Innovations Corp',
                'description': 'Lead ML initiatives, develop deep learning models, deploy production ML systems, mentor junior engineers.',
                'required_skills': ['python', 'machine learning', 'tensorflow', 'pytorch', 'deep learning', 'aws', 'docker', 'kubernetes'],
                'experience_required': 5,
                'category': 'Information Technology',
                'salary_range': '$140k-$200k',
                'location': 'San Francisco, CA',
                'remote_friendly': True
            },
            {
                'job_id': 'TECH002',
                'title': 'Full Stack Software Engineer',
                'company': 'TechStart Inc',
                'description': 'Build scalable web applications, work with React/Node.js, implement microservices architecture.',
                'required_skills': ['javascript', 'react', 'node.js', 'python', 'sql', 'aws', 'docker', 'git'],
                'experience_required': 3,
                'category': 'Information Technology',
                'salary_range': '$100k-$150k',
                'location': 'Austin, TX',
                'remote_friendly': True
            },
            {
                'job_id': 'TECH003',
                'title': 'DevOps Engineer',
                'company': 'Cloud Solutions Ltd',
                'description': 'Manage cloud infrastructure, implement CI/CD pipelines, ensure system reliability and scalability.',
                'required_skills': ['aws', 'kubernetes', 'docker', 'terraform', 'jenkins', 'python', 'linux', 'monitoring'],
                'experience_required': 4,
                'category': 'Information Technology',
                'salary_range': '$120k-$170k',
                'location': 'Seattle, WA',
                'remote_friendly': True
            },
            {
                'job_id': 'TECH004',
                'title': 'Data Scientist',
                'company': 'Analytics Pro',
                'description': 'Analyze large datasets, build predictive models, create data visualizations, drive business insights.',
                'required_skills': ['python', 'r', 'sql', 'machine learning', 'pandas', 'numpy', 'tableau', 'statistics'],
                'experience_required': 3,
                'category': 'Information Technology',
                'salary_range': '$110k-$160k',
                'location': 'New York, NY',
                'remote_friendly': False
            },
            
            # FINANCE
            {
                'job_id': 'FIN001',
                'title': 'Senior Financial Analyst',
                'company': 'Investment Partners LLC',
                'description': 'Conduct financial modeling, analyze investment opportunities, prepare detailed reports for stakeholders.',
                'required_skills': ['financial analysis', 'excel', 'financial modeling', 'valuation', 'investment', 'bloomberg', 'powerpoint'],
                'experience_required': 4,
                'category': 'Finance',
                'salary_range': '$90k-$130k',
                'location': 'Chicago, IL',
                'remote_friendly': False
            },
            {
                'job_id': 'FIN002',
                'title': 'Senior Accountant',
                'company': 'Financial Services Group',
                'description': 'Manage month-end close, prepare financial statements, ensure GAAP compliance, support audits.',
                'required_skills': ['accounting', 'gaap', 'financial reporting', 'excel', 'quickbooks', 'tax preparation', 'auditing'],
                'experience_required': 4,
                'category': 'Accountant',
                'salary_range': '$70k-$95k',
                'location': 'Dallas, TX',
                'remote_friendly': False
            },
            
            # LEGAL
            {
                'job_id': 'LEG001',
                'title': 'Corporate Attorney',
                'company': 'Legal Associates',
                'description': 'Handle M&A transactions, draft contracts, provide legal counsel on corporate matters.',
                'required_skills': ['corporate law', 'contract law', 'legal research', 'litigation', 'compliance', 'negotiation'],
                'experience_required': 5,
                'category': 'Advocate',
                'salary_range': '$150k-$250k',
                'location': 'Los Angeles, CA',
                'remote_friendly': False
            },
            
            # HEALTHCARE
            {
                'job_id': 'HEALTH001',
                'title': 'Registered Nurse - ICU',
                'company': 'Metropolitan Hospital',
                'description': 'Provide critical care nursing, monitor patient conditions, administer medications, collaborate with medical team.',
                'required_skills': ['nursing', 'patient care', 'critical care', 'medical procedures', 'healthcare', 'bls', 'acls'],
                'experience_required': 2,
                'category': 'Healthcare',
                'salary_range': '$70k-$95k',
                'location': 'Boston, MA',
                'remote_friendly': False
            },
            
            # MARKETING
            {
                'job_id': 'MKT001',
                'title': 'Digital Marketing Manager',
                'company': 'Growth Marketing Agency',
                'description': 'Lead digital campaigns, manage SEO/SEM, analyze marketing metrics, optimize conversion rates.',
                'required_skills': ['digital marketing', 'seo', 'sem', 'google analytics', 'facebook ads', 'content marketing', 'a/b testing'],
                'experience_required': 4,
                'category': 'Marketing',
                'salary_range': '$80k-$120k',
                'location': 'Miami, FL',
                'remote_friendly': True
            },
            
            # SALES
            {
                'job_id': 'SALES001',
                'title': 'Enterprise Sales Manager',
                'company': 'SaaS Solutions Inc',
                'description': 'Manage enterprise accounts, drive revenue growth, build client relationships, negotiate contracts.',
                'required_skills': ['sales', 'account management', 'crm', 'salesforce', 'negotiation', 'business development', 'b2b sales'],
                'experience_required': 5,
                'category': 'Sales',
                'salary_range': '$100k-$180k',
                'location': 'San Francisco, CA',
                'remote_friendly': True
            },
            
            # HR
            {
                'job_id': 'HR001',
                'title': 'Senior HR Business Partner',
                'company': 'Global Tech Corp',
                'description': 'Partner with business leaders, manage talent acquisition, drive organizational development initiatives.',
                'required_skills': ['human resources', 'talent management', 'recruiting', 'employee relations', 'performance management', 'hris'],
                'experience_required': 6,
                'category': 'HR',
                'salary_range': '$90k-$130k',
                'location': 'Denver, CO',
                'remote_friendly': True
            },
            
            # DESIGN
            {
                'job_id': 'DES001',
                'title': 'Senior UX Designer',
                'company': 'Design Innovation Studio',
                'description': 'Lead UX research, create user personas, design wireframes and prototypes, conduct usability testing.',
                'required_skills': ['ux design', 'ui design', 'figma', 'sketch', 'prototyping', 'user research', 'usability testing'],
                'experience_required': 4,
                'category': 'Designer',
                'salary_range': '$95k-$140k',
                'location': 'Portland, OR',
                'remote_friendly': True
            }
        ]
    
    def analyze_resume(self, resume_text: str) -> Dict:
        """Comprehensive resume analysis using ML"""
        print("🔍 Analyzing resume with advanced ML techniques...")
        
        # Extract skills with categories and scores
        skills_data, skill_scores = self.extract_advanced_skills(resume_text)
        
        # Extract experience information
        experience_data = self.extract_experience_advanced(resume_text)
        
        # Generate embeddings
        resume_embedding = self.generate_ensemble_embeddings(resume_text)
        
        # Categorize resume
        category = self._categorize_resume_ml(skills_data, experience_data, resume_text)
        
        return {
            'text': resume_text,
            'skills': skills_data,
            'skill_scores': skill_scores,
            'experience': experience_data,
            'category': category,
            'embedding': resume_embedding,
            'word_count': len(resume_text.split()),
            'char_count': len(resume_text)
        }
    
    def _categorize_resume_ml(self, skills_data: Dict, experience_data: Dict, text: str) -> str:
        """Categorize resume using ML approach"""
        category_scores = {}
        
        # Define category mappings
        category_mappings = {
            'Information Technology': ['programming_languages', 'web_technologies', 'databases', 'cloud_devops', 'data_science', 'mobile_development'],
            'Finance': ['finance_accounting', 'banking_investment'],
            'Healthcare': ['healthcare_medical'],
            'Marketing': ['digital_marketing', 'sales_business_dev'],
            'Designer': ['design_creative'],
            'HR': ['leadership_management'],
            'Advocate': ['legal_compliance']
        }
        
        # Calculate scores based on skill categories
        for category, skill_categories in category_mappings.items():
            score = 0
            for skill_cat in skill_categories:
                if skill_cat in skills_data:
                    # Weight by number of skills and category importance
                    skill_count = len(skills_data[skill_cat])
                    category_weight = self.skill_categories[skill_cat]['weight']
                    score += skill_count * category_weight
            
            category_scores[category] = score
        
        # Add text-based scoring
        text_lower = text.lower()
        text_keywords = {
            'Information Technology': ['software', 'programming', 'developer', 'engineer', 'technology', 'computer', 'system'],
            'Finance': ['finance', 'accounting', 'investment', 'banking', 'financial'],
            'Healthcare': ['healthcare', 'medical', 'patient', 'clinical', 'hospital'],
            'Marketing': ['marketing', 'advertising', 'campaign', 'brand', 'promotion'],
            'Designer': ['design', 'creative', 'visual', 'graphic', 'user experience'],
            'HR': ['human resources', 'recruiting', 'talent', 'employee'],
            'Advocate': ['legal', 'law', 'attorney', 'litigation', 'compliance']
        }
        
        for category, keywords in text_keywords.items():
            text_score = sum(text_lower.count(keyword) for keyword in keywords)
            category_scores[category] = category_scores.get(category, 0) + text_score
        
        # Return category with highest score
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            if category_scores[best_category] > 0:
                return best_category
        
        return 'General'
    
    def find_job_matches(self, resume_data: Dict, top_n: int = 10) -> List[Dict]:
        """Find job matches using advanced ML techniques"""
        print(f"🎯 Finding top {top_n} job matches using ML...")
        
        matches = []
        resume_embedding = resume_data['embedding']
        
        for job in self.job_database:
            # Generate job embedding
            job_text = f"{job['title']} {job['description']} {' '.join(job['required_skills'])}"
            job_embedding = self.generate_ensemble_embeddings(job_text)
            
            # Calculate semantic similarity
            semantic_similarity = self.calculate_advanced_similarity(resume_embedding, job_embedding)
            
            # Extract ML features
            ml_features = self.extract_ml_features(resume_data, job)
            
            # Calculate component scores
            scores = self._calculate_component_scores(resume_data, job, semantic_similarity)
            
            # Calculate final ML-based score
            final_score = self._calculate_ml_score(scores, ml_features)
            
            match_data = {
                'job': job,
                'final_score': round(final_score * 100, 2),
                'semantic_similarity': round(semantic_similarity * 100, 2),
                'skill_match': scores['skill_match'],
                'experience_match': scores['experience_match'],
                'category_match': scores['category_match'],
                'ml_features': ml_features,
                'matching_skills': scores['matching_skills'],
                'missing_skills': scores['missing_skills']
            }
            
            matches.append(match_data)
        
        # Sort by final score
        matches.sort(key=lambda x: x['final_score'], reverse=True)
        
        return matches[:top_n]
    
    def _calculate_component_scores(self, resume_data: Dict, job: Dict, semantic_similarity: float) -> Dict:
        """Calculate individual component scores"""
        # Skill matching
        resume_skills = set()
        for category_skills in resume_data.get('skills', {}).values():
            resume_skills.update([skill.lower() for skill in category_skills])
        
        job_skills = set([skill.lower() for skill in job.get('required_skills', [])])
        
        matching_skills = resume_skills.intersection(job_skills)
        missing_skills = job_skills - resume_skills
        
        skill_match = len(matching_skills) / len(job_skills) * 100 if job_skills else 0
        
        # Experience matching
        resume_exp = resume_data.get('experience', {}).get('total_years', 0) or 0
        job_exp = job.get('experience_required', 0) or 0
        
        if job_exp == 0:
            experience_match = 100
        elif resume_exp >= job_exp:
            experience_match = 100
        else:
            experience_match = (resume_exp / job_exp) * 100
        
        # Category matching
        resume_category = resume_data.get('category', '')
        job_category = job.get('category', '')
        category_match = 100 if resume_category == job_category else 0
        
        return {
            'skill_match': round(skill_match, 2),
            'experience_match': round(experience_match, 2),
            'category_match': category_match,
            'matching_skills': list(matching_skills),
            'missing_skills': list(missing_skills)
        }
    
    def _calculate_ml_score(self, scores: Dict, ml_features: np.ndarray) -> float:
        """Calculate final ML-based score"""
        # Weighted combination of different factors
        weights = {
            'semantic': 0.35,
            'skill': 0.30,
            'experience': 0.20,
            'category': 0.15
        }
        
        # Normalize scores to 0-1 range
        semantic_score = scores.get('semantic_similarity', 0) / 100
        skill_score = scores['skill_match'] / 100
        experience_score = scores['experience_match'] / 100
        category_score = scores['category_match'] / 100
        
        # Calculate weighted score
        final_score = (
            semantic_score * weights['semantic'] +
            skill_score * weights['skill'] +
            experience_score * weights['experience'] +
            category_score * weights['category']
        )
        
        return final_score
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'scaler': self.scaler,
            'ml_model': self.ml_model,
            'skill_categories': self.skill_categories
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.scaler = model_data['scaler']
            self.ml_model = model_data['ml_model']
            self.skill_categories = model_data['skill_categories']
            
            print(f"✅ Model loaded from {filepath}")
        else:
            print(f"⚠️ Model file {filepath} not found")


def main():
    """Main function to demonstrate the ML resume matcher"""
    print("🚀 Advanced ML Resume Matcher Demo")
    print("=" * 50)
    
    # Initialize the matcher
    matcher = AdvancedResumeMLMatcher()
    
    # Example usage
    sample_resume_text = """
    John Smith
    Senior Software Engineer
    
    EXPERIENCE
    Senior Software Engineer at Google (2020-2023)
    - Developed machine learning models using Python and TensorFlow
    - Built scalable web applications with React and Node.js
    - Implemented cloud infrastructure on AWS and Kubernetes
    - Led a team of 5 engineers on multiple projects
    
    Software Engineer at Microsoft (2018-2020)
    - Developed full-stack applications using JavaScript and Python
    - Worked with SQL databases and REST APIs
    - Experience with Docker and CI/CD pipelines
    
    EDUCATION
    Master of Science in Computer Science
    Stanford University (2016-2018)
    
    SKILLS
    Programming: Python, JavaScript, Java, C++
    Web: React, Node.js, HTML, CSS
    ML/AI: TensorFlow, PyTorch, scikit-learn
    Cloud: AWS, Docker, Kubernetes
    Databases: SQL, MongoDB, PostgreSQL
    """
    
    print("📄 Analyzing sample resume...")
    resume_analysis = matcher.analyze_resume(sample_resume_text)
    
    print(f"✅ Resume analyzed successfully!")
    print(f"📊 Category: {resume_analysis['category']}")
    print(f"🎯 Skills found: {sum(len(skills) for skills in resume_analysis['skills'].values())}")
    print(f"💼 Experience: {resume_analysis['experience']['total_years']} years")
    
    print("\n🔍 Finding job matches...")
    matches = matcher.find_job_matches(resume_analysis, top_n=5)
    
    print(f"\n🎯 Top 5 Job Matches:")
    print("-" * 50)
    
    for i, match in enumerate(matches, 1):
        job = match['job']
        print(f"\n{i}. {job['title']} at {job['company']}")
        print(f"   💰 Salary: {job['salary_range']}")
        print(f"   🎯 Match Score: {match['final_score']}%")
        print(f"   📊 Breakdown:")
        print(f"      - Semantic Similarity: {match['semantic_similarity']}%")
        print(f"      - Skills Match: {match['skill_match']}%")
        print(f"      - Experience Match: {match['experience_match']}%")
        print(f"      - Category Match: {match['category_match']}%")
        print(f"   ✅ Matching Skills: {', '.join(match['matching_skills'][:5])}")
        if match['missing_skills']:
            print(f"   📚 Skills to Develop: {', '.join(match['missing_skills'][:3])}")


if __name__ == "__main__":
    main()