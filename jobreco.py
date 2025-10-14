import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import json

class UniversalJobRecommendationEngine:
    def __init__(self, resume_data_path='enhanced_resume_data.csv', 
                 embeddings_path='resume_embeddings.npy'):
        """Initialize the recommendation engine"""
        self.resume_df = pd.read_csv(resume_data_path)
        self.embeddings = np.load(embeddings_path)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Universal job database across industries
        self.job_database = self.create_universal_job_database()
        self.job_embeddings = self.generate_job_embeddings()
    
    def create_universal_job_database(self):
        """Create a diverse job database across multiple industries"""
        jobs = [
            # Business & Management
            {
                'job_id': 'JOB001',
                'title': 'Business Development Manager',
                'company': 'Global Corp',
                'description': 'Seeking experienced professional to drive business growth, manage client relationships, and lead sales initiatives. Strong communication and strategic planning required.',
                'required_skills': ['sales', 'business development', 'communication', 'negotiation', 'leadership', 'client relations'],
                'experience_required': 5,
                'category': 'Sales & Marketing',
                'salary_range': '$60k-$90k',
                'industry': 'Business'
            },
            # Finance & Accounting
            {
                'job_id': 'JOB002',
                'title': 'Financial Analyst',
                'company': 'Finance Solutions Inc',
                'description': 'Analyze financial data, prepare reports, and provide insights for business decisions. Excel and financial modeling skills essential.',
                'required_skills': ['accounting', 'financial analysis', 'excel', 'budgeting', 'analytical', 'reporting'],
                'experience_required': 3,
                'category': 'Finance & Accounting',
                'salary_range': '$55k-$80k',
                'industry': 'Finance'
            },
            # Healthcare
            {
                'job_id': 'JOB003',
                'title': 'Registered Nurse',
                'company': 'City Hospital',
                'description': 'Provide patient care, administer medications, and work with healthcare team. Valid nursing license required.',
                'required_skills': ['patient care', 'medical', 'nursing', 'healthcare', 'clinical', 'communication'],
                'experience_required': 2,
                'category': 'Healthcare',
                'salary_range': '$60k-$85k',
                'industry': 'Healthcare'
            },
            # Education
            {
                'job_id': 'JOB004',
                'title': 'High School Teacher',
                'company': 'Lincoln High School',
                'description': 'Teach students, develop curriculum, assess learning outcomes. Teaching certification required.',
                'required_skills': ['teaching', 'education', 'curriculum', 'classroom management', 'communication', 'patience'],
                'experience_required': 2,
                'category': 'Education',
                'salary_range': '$45k-$65k',
                'industry': 'Education'
            },
            # Human Resources
            {
                'job_id': 'JOB005',
                'title': 'HR Manager',
                'company': 'Corporate Solutions',
                'description': 'Manage recruitment, employee relations, training programs, and HR policies. Strong interpersonal skills needed.',
                'required_skills': ['recruitment', 'human resources', 'employee relations', 'communication', 'organization', 'leadership'],
                'experience_required': 4,
                'category': 'Human Resources',
                'salary_range': '$55k-$75k',
                'industry': 'Human Resources'
            },
            # Marketing & Communications
            {
                'job_id': 'JOB006',
                'title': 'Marketing Coordinator',
                'company': 'Creative Agency',
                'description': 'Develop marketing campaigns, manage social media, coordinate events. Creative and organized individual needed.',
                'required_skills': ['marketing', 'communication', 'creativity', 'social media', 'organization', 'branding'],
                'experience_required': 2,
                'category': 'Marketing',
                'salary_range': '$45k-$65k',
                'industry': 'Marketing'
            },
            # Operations & Logistics
            {
                'job_id': 'JOB007',
                'title': 'Operations Manager',
                'company': 'Supply Chain Co',
                'description': 'Oversee daily operations, manage logistics, improve processes. Strong problem-solving and leadership skills required.',
                'required_skills': ['operations', 'logistics', 'supply chain', 'leadership', 'problem solving', 'process improvement'],
                'experience_required': 5,
                'category': 'Operations',
                'salary_range': '$65k-$95k',
                'industry': 'Operations'
            },
            # Customer Service
            {
                'job_id': 'JOB008',
                'title': 'Customer Service Representative',
                'company': 'ServiceFirst Inc',
                'description': 'Handle customer inquiries, resolve issues, maintain satisfaction. Excellent communication and problem-solving required.',
                'required_skills': ['customer service', 'communication', 'problem solving', 'patience', 'interpersonal', 'typing'],
                'experience_required': 1,
                'category': 'Customer Service',
                'salary_range': '$35k-$50k',
                'industry': 'Service'
            },
            # Administrative
            {
                'job_id': 'JOB009',
                'title': 'Executive Assistant',
                'company': 'Executive Office',
                'description': 'Provide administrative support, manage schedules, coordinate meetings. Strong organizational and communication skills needed.',
                'required_skills': ['administration', 'organization', 'communication', 'microsoft office', 'scheduling', 'coordination'],
                'experience_required': 3,
                'category': 'Administrative',
                'salary_range': '$45k-$65k',
                'industry': 'Administration'
            },
            # Legal
            {
                'job_id': 'JOB010',
                'title': 'Legal Assistant',
                'company': 'Law Associates',
                'description': 'Support attorneys with research, document preparation, and case management. Legal knowledge and attention to detail required.',
                'required_skills': ['legal', 'research', 'writing', 'attention to detail', 'organization', 'communication'],
                'experience_required': 2,
                'category': 'Legal',
                'salary_range': '$40k-$60k',
                'industry': 'Legal'
            },
            # Sales
            {
                'job_id': 'JOB011',
                'title': 'Sales Representative',
                'company': 'Retail Solutions',
                'description': 'Generate leads, close sales, build client relationships. Strong persuasion and communication skills essential.',
                'required_skills': ['sales', 'communication', 'persuasion', 'negotiation', 'customer service', 'lead generation'],
                'experience_required': 2,
                'category': 'Sales',
                'salary_range': '$40k-$70k',
                'industry': 'Sales'
            },
            # Project Management
            {
                'job_id': 'JOB012',
                'title': 'Project Coordinator',
                'company': 'Project Pros',
                'description': 'Coordinate projects, manage timelines, communicate with stakeholders. Organization and multitasking abilities required.',
                'required_skills': ['project management', 'organization', 'communication', 'planning', 'coordination', 'teamwork'],
                'experience_required': 3,
                'category': 'Project Management',
                'salary_range': '$50k-$70k',
                'industry': 'Management'
            }
        ]
        return pd.DataFrame(jobs)
    
    def generate_job_embeddings(self):
        """Generate embeddings for job descriptions"""
        job_texts = (self.job_database['title'] + ' ' + 
                    self.job_database['description'] + ' ' + 
                    self.job_database['required_skills'].apply(lambda x: ' '.join(x)))
        
        embeddings = self.embedding_model.encode(job_texts.tolist())
        return embeddings
    
    def calculate_skill_match_score(self, resume_skills, job_skills):
        """Calculate skill match percentage"""
        if not resume_skills or not job_skills:
            return 0
        
        # Parse skills if stored as strings
        if isinstance(resume_skills, str):
            resume_skills = eval(resume_skills) if resume_skills.startswith('[') else []
        if isinstance(job_skills, str):
            job_skills = eval(job_skills) if job_skills.startswith('[') else []
        
        resume_skills_set = set([s.lower() for s in resume_skills])
        job_skills_set = set([s.lower() for s in job_skills])
        
        if not job_skills_set:
            return 0
        
        matched_skills = resume_skills_set.intersection(job_skills_set)
        return len(matched_skills) / len(job_skills_set) * 100
    
    def calculate_experience_match(self, resume_exp, job_exp):
        """Calculate experience match score"""
        if resume_exp is None or pd.isna(resume_exp):
            return 50  # Neutral score
        
        if resume_exp >= job_exp:
            return 100
        else:
            return (resume_exp / job_exp) * 100
    
    def calculate_semantic_similarity(self, resume_embedding, job_embedding):
        """Calculate cosine similarity"""
        if resume_embedding is None:
            return 0
        
        resume_emb = resume_embedding.reshape(1, -1)
        job_emb = job_embedding.reshape(1, -1)
        
        similarity = cosine_similarity(resume_emb, job_emb)[0][0]
        return similarity * 100
    
    def get_recommendations_for_resume(self, resume_id, top_n=5):
        """Get top N job recommendations for a specific resume"""
        resume_idx = self.resume_df[self.resume_df['resume_id'] == resume_id].index
        
        if len(resume_idx) == 0:
            return None
        
        resume_idx = resume_idx[0]
        resume = self.resume_df.iloc[resume_idx]
        resume_embedding = self.embeddings[resume_idx]
        
        recommendations = []
        
        for job_idx, job in self.job_database.iterrows():
            semantic_score = self.calculate_semantic_similarity(
                resume_embedding, 
                self.job_embeddings[job_idx]
            )
            
            skill_score = self.calculate_skill_match_score(
                resume['skills'],
                job['required_skills']
            )
            
            exp_score = self.calculate_experience_match(
                resume['experience_years'],
                job['experience_required']
            )
            
            # Weighted final score
            final_score = (
                semantic_score * 0.4 + 
                skill_score * 0.4 + 
                exp_score * 0.2
            )
            
            recommendations.append({
                'job_id': job['job_id'],
                'title': job['title'],
                'company': job['company'],
                'category': job['category'],
                'industry': job['industry'],
                'salary_range': job['salary_range'],
                'match_score': round(final_score, 2),
                'semantic_similarity': round(semantic_score, 2),
                'skill_match': round(skill_score, 2),
                'experience_match': round(exp_score, 2),
                'required_skills': job['required_skills'],
                'description': job['description']
            })
        
        recommendations = sorted(recommendations, key=lambda x: x['match_score'], reverse=True)
        
        return {
            'resume_id': resume_id,
            'candidate_skills': resume['skills'],
            'candidate_experience': resume['experience_years'],
            'candidate_category': resume['category'],
            'recommendations': recommendations[:top_n]
        }
    
    def get_candidates_for_job(self, job_id, top_n=10):
        """Get top candidates for a specific job"""
        job = self.job_database[self.job_database['job_id'] == job_id]
        
        if len(job) == 0:
            return None
        
        job = job.iloc[0]
        job_idx = self.job_database[self.job_database['job_id'] == job_id].index[0]
        job_embedding = self.job_embeddings[job_idx]
        
        candidates = []
        
        for idx, resume in self.resume_df.iterrows():
            resume_embedding = self.embeddings[idx]
            
            semantic_score = self.calculate_semantic_similarity(
                resume_embedding,
                job_embedding
            )
            
            skill_score = self.calculate_skill_match_score(
                resume['skills'],
                job['required_skills']
            )
            
            exp_score = self.calculate_experience_match(
                resume['experience_years'],
                job['experience_required']
            )
            
            final_score = (
                semantic_score * 0.4 + 
                skill_score * 0.4 + 
                exp_score * 0.2
            )
            
            candidates.append({
                'resume_id': resume['resume_id'],
                'category': resume['category'],
                'match_score': round(final_score, 2),
                'semantic_similarity': round(semantic_score, 2),
                'skill_match': round(skill_score, 2),
                'experience_match': round(exp_score, 2),
                'candidate_skills': resume['skills'],
                'experience_years': resume['experience_years']
            })
        
        candidates = sorted(candidates, key=lambda x: x['match_score'], reverse=True)
        
        return {
            'job_id': job_id,
            'job_title': job['title'],
            'company': job['company'],
            'industry': job['industry'],
            'required_skills': job['required_skills'],
            'top_candidates': candidates[:top_n]
        }


# Usage Example
if __name__ == "__main__":
    engine = UniversalJobRecommendationEngine()
    
    print("=" * 80)
    print("UNIVERSAL JOB RECOMMENDATION SYSTEM")
    print("=" * 80)
    
    # Example 1: Get recommendations for a resume
    sample_resume_id = engine.resume_df.iloc[0]['resume_id']
    recommendations = engine.get_recommendations_for_resume(sample_resume_id, top_n=5)
    
    if recommendations:
        print(f"\nResume ID: {recommendations['resume_id']}")
        print(f"Category: {recommendations['candidate_category']}")
        print(f"Experience: {recommendations['candidate_experience']} years\n")
        
        print("Top 5 Job Recommendations:")
        for i, rec in enumerate(recommendations['recommendations'], 1):
            print(f"\n{i}. {rec['title']} at {rec['company']}")
            print(f"   Industry: {rec['industry']}")
            print(f"   Match Score: {rec['match_score']}%")
            print(f"   Salary: {rec['salary_range']}")
    
    # Save all recommendations
    all_recs = []
    for idx in range(min(100, len(engine.resume_df))):
        resume_id = engine.resume_df.iloc[idx]['resume_id']
        rec = engine.get_recommendations_for_resume(resume_id, top_n=3)
        if rec:
            all_recs.append(rec)
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif pd.isna(obj):
            return None
        return obj
    
    all_recs_converted = convert_to_native(all_recs)
    
    with open('universal_job_recommendations.json', 'w') as f:
        json.dump(all_recs_converted, f, indent=2)
    
    print("\n\nRecommendations saved to 'universal_job_recommendations.json'")