from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import re
from collections import Counter

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def extract_text_from_pdf(pdf_path):
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

def extract_comprehensive_skills(text):
    """Extract comprehensive skills from text"""
    skill_categories = {
        # TECHNOLOGY & IT
        'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin', 'go', 'programming', 'coding'],
        'web': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'web development', 'frontend', 'backend'],
        'data': ['sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'database', 'data analysis', 'excel'],
        'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'cloud computing', 'devops'],
        
        # FINANCE & ACCOUNTING
        'finance': ['financial analysis', 'investment', 'portfolio management', 'budgeting', 'forecasting', 'finance'],
        'accounting': ['accounting', 'bookkeeping', 'tax preparation', 'auditing', 'financial reporting', 'gaap', 'quickbooks'],
        'banking': ['banking', 'loans', 'credit analysis', 'relationship management', 'financial products'],
        
        # LEGAL
        'legal': ['legal research', 'contract law', 'corporate law', 'litigation', 'legal writing', 'law', 'attorney', 'lawyer'],
        
        # HEALTHCARE
        'healthcare': ['patient care', 'nursing', 'medical procedures', 'clinical skills', 'healthcare', 'medical'],
        
        # EDUCATION
        'education': ['teaching', 'curriculum development', 'lesson planning', 'classroom management', 'education'],
        
        # SALES & MARKETING
        'sales': ['sales management', 'lead generation', 'negotiation', 'crm', 'sales', 'business development'],
        'marketing': ['digital marketing', 'social media', 'seo', 'content marketing', 'marketing', 'advertising'],
        
        # HUMAN RESOURCES
        'hr': ['human resources', 'recruiting', 'talent acquisition', 'employee relations', 'hr'],
        
        # ENGINEERING
        'engineering': ['mechanical engineering', 'civil engineering', 'electrical engineering', 'cad', 'autocad', 'engineering'],
        
        # DESIGN & CREATIVE
        'design': ['graphic design', 'ux design', 'ui design', 'visual design', 'design', 'creative'],
        
        # OTHER INDUSTRIES
        'culinary': ['culinary arts', 'cooking', 'food preparation', 'menu planning', 'chef', 'kitchen management'],
        'construction': ['construction management', 'project management', 'safety management', 'construction'],
        'agriculture': ['farming', 'crop management', 'agriculture', 'agricultural'],
        'fitness': ['personal training', 'fitness coaching', 'nutrition', 'fitness'],
        'consulting': ['business consulting', 'strategy consulting', 'consulting'],
        'aviation': ['flight operations', 'aviation', 'pilot'],
        
        # SOFT SKILLS
        'leadership': ['leadership', 'team management', 'mentoring', 'management'],
        'communication': ['communication', 'presentation', 'public speaking', 'interpersonal'],
        'analytical': ['analytical thinking', 'problem solving', 'research', 'analysis']
    }
    
    text_lower = text.lower()
    found_skills = []
    
    for category, skills in skill_categories.items():
        for skill in skills:
            if skill in text_lower:
                found_skills.append(skill)
    
    return list(set(found_skills))

def extract_experience(text):
    """Extract years of experience"""
    patterns = [
        r'(\d+)\+?\s*years?\s*of\s*experience',
        r'experience\s*[:\-]?\s*(\d+)\+?\s*years?',
        r'(\d+)\+?\s*years?\s*experience'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return int(match.group(1))
    
    return None

def categorize_resume(text, skills):
    """Categorize resume based on content and skills"""
    text_lower = text.lower()
    
    category_keywords = {
        'Information Technology': ['software', 'programming', 'developer', 'engineer', 'technology', 'it', 'computer', 'web', 'mobile', 'database'],
        'Accountant': ['accountant', 'accounting', 'bookkeeper', 'financial reporting', 'tax', 'audit', 'cpa'],
        'Finance': ['finance', 'financial analyst', 'investment', 'banking', 'portfolio', 'risk management'],
        'Banking': ['bank', 'banking', 'loan officer', 'credit', 'mortgage', 'financial services'],
        'Advocate': ['lawyer', 'attorney', 'legal', 'law', 'litigation', 'counsel', 'paralegal'],
        'Healthcare': ['nurse', 'doctor', 'physician', 'medical', 'healthcare', 'clinical', 'hospital'],
        'Teacher': ['teacher', 'educator', 'professor', 'instructor', 'education', 'teaching', 'academic'],
        'Sales': ['sales', 'account manager', 'business development', 'sales representative'],
        'Digital Media': ['marketing', 'brand', 'advertising', 'digital marketing', 'social media'],
        'HR': ['human resources', 'hr', 'recruiter', 'talent acquisition', 'employee relations'],
        'Engineering': ['engineer', 'engineering', 'mechanical', 'civil', 'electrical'],
        'Designer': ['designer', 'design', 'graphic', 'creative', 'ux', 'ui', 'visual'],
        'Chef': ['chef', 'cook', 'culinary', 'kitchen', 'restaurant', 'food service'],
        'Construction': ['construction', 'contractor', 'builder', 'project manager', 'site supervisor'],
        'Agriculture': ['agriculture', 'farming', 'agricultural', 'crop', 'livestock'],
        'Fitness': ['fitness', 'trainer', 'coach', 'gym', 'exercise', 'wellness'],
        'Consultant': ['consultant', 'consulting', 'advisor', 'strategy'],
        'Aviation': ['pilot', 'aviation', 'aircraft', 'flight', 'airline']
    }
    
    category_scores = {}
    for category, keywords in category_keywords.items():
        score = 0
        for keyword in keywords:
            if keyword in text_lower:
                score += text_lower.count(keyword)
        
        # Boost score if skills match category
        skill_boost = sum(1 for skill in skills if any(keyword in skill.lower() for keyword in keywords))
        category_scores[category] = score + (skill_boost * 2)
    
    if category_scores:
        best_category = max(category_scores, key=category_scores.get)
        if category_scores[best_category] > 0:
            return best_category
    
    return 'General'

# Comprehensive job database
JOBS = [
    # TECHNOLOGY
    {'title': 'Senior Python Developer', 'company': 'Tech Corp', 'skills': ['python', 'django', 'flask', 'aws', 'sql', 'programming'], 'salary': '$100k-$140k', 'category': 'Information Technology', 'description': 'Develop scalable web applications using Python frameworks and cloud technologies.'},
    {'title': 'Machine Learning Engineer', 'company': 'AI Innovations', 'skills': ['python', 'machine learning', 'tensorflow', 'data analysis', 'programming'], 'salary': '$120k-$160k', 'category': 'Information Technology', 'description': 'Build and deploy ML models for production systems.'},
    {'title': 'Full Stack Developer', 'company': 'Web Solutions', 'skills': ['javascript', 'react', 'node.js', 'html', 'css', 'sql'], 'salary': '$90k-$130k', 'category': 'Information Technology', 'description': 'Develop end-to-end web applications with modern frameworks.'},
    
    # FINANCE & ACCOUNTING
    {'title': 'Senior Accountant', 'company': 'Financial Services LLC', 'skills': ['accounting', 'financial reporting', 'tax preparation', 'excel', 'gaap'], 'salary': '$65k-$85k', 'category': 'Accountant', 'description': 'Handle complex accounting tasks, financial reporting, and tax compliance.'},
    {'title': 'Financial Analyst', 'company': 'Investment Group', 'skills': ['financial analysis', 'excel', 'investment', 'budgeting', 'finance'], 'salary': '$70k-$95k', 'category': 'Finance', 'description': 'Analyze financial data and provide investment recommendations.'},
    {'title': 'Banking Relationship Manager', 'company': 'First National Bank', 'skills': ['banking', 'relationship management', 'sales', 'financial products'], 'salary': '$60k-$80k', 'category': 'Banking', 'description': 'Manage client relationships and develop banking solutions.'},
    
    # LEGAL
    {'title': 'Corporate Lawyer', 'company': 'Legal Associates', 'skills': ['legal research', 'contract law', 'corporate law', 'litigation'], 'salary': '$120k-$180k', 'category': 'Advocate', 'description': 'Handle corporate legal matters and contract negotiations.'},
    {'title': 'Legal Assistant', 'company': 'Law Firm Partners', 'skills': ['legal research', 'legal writing', 'case management'], 'salary': '$45k-$60k', 'category': 'Advocate', 'description': 'Support attorneys with research and document preparation.'},
    
    # HEALTHCARE
    {'title': 'Registered Nurse', 'company': 'City Medical Center', 'skills': ['patient care', 'nursing', 'medical procedures', 'healthcare'], 'salary': '$65k-$85k', 'category': 'Healthcare', 'description': 'Provide comprehensive patient care and medical support.'},
    {'title': 'Healthcare Administrator', 'company': 'Regional Hospital', 'skills': ['healthcare', 'management', 'budgeting', 'leadership'], 'salary': '$75k-$105k', 'category': 'Healthcare', 'description': 'Manage healthcare operations and administrative functions.'},
    
    # EDUCATION
    {'title': 'High School Mathematics Teacher', 'company': 'Lincoln High School', 'skills': ['teaching', 'education', 'curriculum development', 'classroom management'], 'salary': '$45k-$65k', 'category': 'Teacher', 'description': 'Teach mathematics and develop engaging lesson plans.'},
    {'title': 'Elementary School Teacher', 'company': 'Sunshine Elementary', 'skills': ['teaching', 'education', 'classroom management', 'communication'], 'salary': '$42k-$62k', 'category': 'Teacher', 'description': 'Educate elementary students across multiple subjects.'},
    
    # SALES & MARKETING
    {'title': 'Sales Manager', 'company': 'Enterprise Solutions', 'skills': ['sales management', 'leadership', 'business development', 'crm'], 'salary': '$70k-$100k', 'category': 'Sales', 'description': 'Lead sales team and develop revenue growth strategies.'},
    {'title': 'Digital Marketing Specialist', 'company': 'Creative Agency', 'skills': ['digital marketing', 'social media', 'seo', 'content marketing'], 'salary': '$50k-$70k', 'category': 'Digital Media', 'description': 'Create and execute digital marketing campaigns.'},
    
    # HUMAN RESOURCES
    {'title': 'HR Business Partner', 'company': 'Global Corporation', 'skills': ['human resources', 'talent management', 'employee relations', 'leadership'], 'salary': '$80k-$110k', 'category': 'HR', 'description': 'Partner with business leaders on HR strategy and talent development.'},
    {'title': 'Recruiter', 'company': 'Talent Acquisition Inc', 'skills': ['recruiting', 'talent acquisition', 'hr', 'communication'], 'salary': '$55k-$75k', 'category': 'HR', 'description': 'Source and recruit top talent for various positions.'},
    
    # ENGINEERING
    {'title': 'Mechanical Engineer', 'company': 'Manufacturing Corp', 'skills': ['mechanical engineering', 'cad', 'design', 'project management'], 'salary': '$75k-$105k', 'category': 'Engineering', 'description': 'Design mechanical systems and oversee manufacturing processes.'},
    {'title': 'Civil Engineer', 'company': 'Infrastructure Solutions', 'skills': ['civil engineering', 'autocad', 'project management', 'construction'], 'salary': '$70k-$95k', 'category': 'Engineering', 'description': 'Plan and design infrastructure and construction projects.'},
    
    # DESIGN
    {'title': 'UX/UI Designer', 'company': 'Tech Startup', 'skills': ['ux design', 'ui design', 'design', 'creative'], 'salary': '$70k-$95k', 'category': 'Designer', 'description': 'Design user interfaces and experiences for digital products.'},
    {'title': 'Graphic Designer', 'company': 'Design Studio', 'skills': ['graphic design', 'visual design', 'creative', 'design'], 'salary': '$45k-$65k', 'category': 'Designer', 'description': 'Create visual designs for marketing and branding materials.'},
    
    # CULINARY
    {'title': 'Executive Chef', 'company': 'Fine Dining Restaurant', 'skills': ['culinary arts', 'menu planning', 'kitchen management', 'leadership'], 'salary': '$60k-$85k', 'category': 'Chef', 'description': 'Lead kitchen operations and develop innovative menus.'},
    {'title': 'Sous Chef', 'company': 'Hotel Restaurant', 'skills': ['cooking', 'food preparation', 'culinary arts'], 'salary': '$45k-$60k', 'category': 'Chef', 'description': 'Assist head chef and manage food preparation operations.'},
    
    # CONSTRUCTION
    {'title': 'Construction Project Manager', 'company': 'BuildRight Construction', 'skills': ['project management', 'construction', 'leadership', 'budgeting'], 'salary': '$75k-$105k', 'category': 'Construction', 'description': 'Manage construction projects from planning to completion.'},
    {'title': 'Site Supervisor', 'company': 'Metro Construction', 'skills': ['construction', 'safety management', 'leadership'], 'salary': '$55k-$75k', 'category': 'Construction', 'description': 'Supervise construction activities and ensure safety compliance.'},
    
    # OTHER INDUSTRIES
    {'title': 'Agricultural Specialist', 'company': 'AgriTech Solutions', 'skills': ['agriculture', 'crop management', 'farming', 'research'], 'salary': '$50k-$70k', 'category': 'Agriculture', 'description': 'Provide expertise in sustainable farming and crop management.'},
    {'title': 'Personal Trainer', 'company': 'Elite Fitness Center', 'skills': ['fitness', 'personal training', 'nutrition', 'communication'], 'salary': '$35k-$55k', 'category': 'Fitness', 'description': 'Provide personalized fitness training and wellness coaching.'},
    {'title': 'Management Consultant', 'company': 'Strategic Advisors', 'skills': ['consulting', 'business analysis', 'strategy', 'analytical'], 'salary': '$90k-$130k', 'category': 'Consultant', 'description': 'Provide strategic advice and business process improvements.'},
    {'title': 'Commercial Pilot', 'company': 'Regional Airlines', 'skills': ['aviation', 'flight operations', 'communication'], 'salary': '$80k-$120k', 'category': 'Aviation', 'description': 'Operate commercial aircraft and ensure passenger safety.'}
]

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Resume Matcher</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif; background-color: #0d1117; color: #e6edf3; min-height: 100vh; padding: 0; }
            .container { max-width: 1200px; margin: 0 auto; background-color: #0d1117; }
            .header { background-color: #161b22; color: #f0f6fc; padding: 60px 40px; text-align: center; border-bottom: 1px solid #30363d; }
            .header h1 { font-size: 3rem; margin-bottom: 16px; font-weight: 600; }
            .content { padding: 40px; }
            .upload-area { border: 3px dashed #ddd; border-radius: 15px; padding: 60px 20px; margin: 20px 0; text-align: center; transition: all 0.3s ease; }
            .upload-area:hover { border-color: #667eea; background: #f8f9ff; }
            input[type="file"] { margin: 20px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
            .btn { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 15px 40px; border-radius: 50px; font-size: 1.1em; cursor: pointer; transition: all 0.3s ease; }
            .btn:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4); }
            .features { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 30px; }
            .feature { text-align: center; padding: 20px; border-radius: 15px; background: #f8f9ff; }
            .feature-icon { font-size: 2.5em; margin-bottom: 15px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 AI Resume Matcher</h1>
                <p>Upload your PDF resume and get personalized job recommendations with detailed analytics</p>
            </div>
            
            <div class="content">
                <form action="/upload" method="post" enctype="multipart/form-data">
                    <div class="upload-area">
                        <div style="font-size: 4em; color: #ddd; margin-bottom: 20px;">📄</div>
                        <h3>Upload Your Resume</h3>
                        <input type="file" name="resume" accept=".pdf" required>
                        <br><br>
                        <button type="submit" class="btn">Analyze Resume & Get Job Matches</button>
                    </div>
                </form>
                
                <div class="features">
                    <div class="feature">
                        <div class="feature-icon">🧠</div>
                        <h3>AI-Powered Analysis</h3>
                        <p>Advanced algorithms analyze your resume content and match it with relevant opportunities</p>
                    </div>
                    <div class="feature">
                        <div class="feature-icon">📊</div>
                        <h3>Detailed Scoring</h3>
                        <p>Get comprehensive scores based on skills match, experience fit, and category alignment</p>
                    </div>
                    <div class="feature">
                        <div class="feature-icon">🚀</div>
                        <h3>Career Insights</h3>
                        <p>Receive personalized suggestions to improve your resume and advance your career</p>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    ''')

def render_template_string(template_str):
    """Simple template renderer"""
    return template_str

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'resume' not in request.files:
        flash('No file selected')
        return redirect(url_for('index'))
    
    file = request.files['resume']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    if file and file.filename.lower().endswith('.pdf'):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Extract and analyze
            text = extract_text_from_pdf(filepath)
            if not text:
                flash('Could not extract text from PDF')
                return redirect(url_for('index'))
            
            skills = extract_comprehensive_skills(text)
            experience = extract_experience(text)
            category = categorize_resume(text, skills)
            
            # Find matching jobs with enhanced scoring
            job_matches = []
            for job in JOBS:
                matching_skills = set(skills) & set(job['skills'])
                skill_score = len(matching_skills) / len(job['skills']) * 100 if job['skills'] else 0
                
                # Category bonus
                category_bonus = 20 if job.get('category') == category else 0
                
                # Final score
                final_score = min(skill_score + category_bonus, 100)
                
                job_matches.append({
                    'job': job,
                    'score': round(final_score, 1),
                    'skill_score': round(skill_score, 1),
                    'category_bonus': category_bonus,
                    'matching_skills': list(matching_skills),
                    'missing_skills': list(set(job['skills']) - set(skills))
                })
            
            # Sort by score
            job_matches.sort(key=lambda x: x['score'], reverse=True)
            
            # Clean up
            os.remove(filepath)
            
            # Generate results HTML
            results_html = generate_results_html(skills, experience, category, text, job_matches)
            return results_html
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('index'))
    
    else:
        flash('Please upload a PDF file')
        return redirect(url_for('index'))

def generate_results_html(skills, experience, category, text, job_matches):
    """Generate the results HTML"""
    
    # Generate skill tags
    skill_tags = ''.join([f'<span class="skill-tag">{skill}</span>' for skill in skills[:15]])
    if len(skills) > 15:
        skill_tags += f'<span style="color: #666;">... and {len(skills) - 15} more</span>'
    
    # Generate job cards
    job_cards = ''
    for i, match in enumerate(job_matches[:8], 1):
        job = match['job']
        
        matching_skills_html = ''.join([f'<span class="skill-tag">{skill}</span>' for skill in match['matching_skills']])
        if not matching_skills_html:
            matching_skills_html = '<span style="color: #666;">None found</span>'
        
        missing_skills_html = ''.join([f'<span class="missing-tag">{skill}</span>' for skill in match['missing_skills']])
        
        category_bonus_html = f'<div style="text-align: center; background: #e8f5e8; padding: 10px; border-radius: 5px;"><div style="font-weight: bold; color: #28a745;">+{match["category_bonus"]}%</div><div style="font-size: 0.8em;">Category Match</div></div>' if match['category_bonus'] > 0 else ''
        
        job_cards += f'''
        <div class="job-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <div>
                    <h4>{job['title']}</h4>
                    <p style="color: #666;">{job['company']} • {job['salary']}</p>
                    <p style="color: #888; font-size: 0.9em;">📂 {job.get('category', 'General')}</p>
                </div>
                <div class="score">{match['score']}%</div>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; margin: 15px 0;">
                <div style="text-align: center; background: #f8f9fa; padding: 10px; border-radius: 5px;">
                    <div style="font-weight: bold; color: #007bff;">{match['skill_score']}%</div>
                    <div style="font-size: 0.8em;">Skills Match</div>
                </div>
                {category_bonus_html}
            </div>
            
            <p style="margin: 15px 0;">{job['description']}</p>
            
            <div class="skills">
                <strong>✅ Matching Skills ({len(match['matching_skills'])}):</strong><br>
                {matching_skills_html}
            </div>
            
            {f'<div class="skills"><strong>📚 Skills to Develop ({len(match["missing_skills"])}):</strong><br>{missing_skills_html}</div>' if match['missing_skills'] else ''}
        </div>
        '''
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Resume Analysis Results</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); overflow: hidden; }}
            .header {{ background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 30px 40px; text-align: center; }}
            .header h1 {{ font-size: 2.2em; margin-bottom: 10px; font-weight: 300; }}
            .content {{ padding: 40px; }}
            .summary {{ background: #f8f9ff; border-radius: 15px; padding: 30px; margin-bottom: 40px; border-left: 5px solid #667eea; }}
            .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 20px; margin: 20px 0; }}
            .summary-item {{ text-align: center; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }}
            .summary-number {{ font-size: 2em; font-weight: bold; color: #667eea; margin-bottom: 5px; }}
            .summary-label {{ color: #666; font-size: 0.9em; }}
            .job-card {{ background: white; border-radius: 15px; padding: 25px; margin-bottom: 25px; box-shadow: 0 5px 20px rgba(0,0,0,0.08); border-left: 5px solid #4CAF50; transition: transform 0.2s ease; }}
            .job-card:hover {{ transform: translateY(-2px); }}
            .score {{ font-size: 1.5em; font-weight: bold; color: #28a745; }}
            .skills {{ margin: 15px 0; }}
            .skill-tag {{ background: #007bff; color: white; padding: 5px 12px; border-radius: 20px; font-size: 0.85em; margin: 3px; display: inline-block; }}
            .missing-tag {{ background: #dc3545; color: white; padding: 5px 12px; border-radius: 20px; font-size: 0.85em; margin: 3px; display: inline-block; }}
            .back-btn {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 30px; border-radius: 25px; text-decoration: none; display: inline-block; margin-bottom: 30px; transition: all 0.3s ease; }}
            .back-btn:hover {{ transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3); }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Resume Analysis Results</h1>
                <p>Your personalized job recommendations and career insights</p>
            </div>
            
            <div class="content">
                <a href="/" class="back-btn">← Upload Another Resume</a>
                
                <div class="summary">
                    <h2>📋 Resume Summary</h2>
                    <div class="summary-grid">
                        <div class="summary-item">
                            <div class="summary-number">{len(skills)}</div>
                            <div class="summary-label">Skills Identified</div>
                        </div>
                        <div class="summary-item">
                            <div class="summary-number">{experience if experience else 'N/A'}</div>
                            <div class="summary-label">Years Experience</div>
                        </div>
                        <div class="summary-item">
                            <div class="summary-number">{len(text.split())}</div>
                            <div class="summary-label">Words</div>
                        </div>
                    </div>
                    
                    <div style="margin: 20px 0;">
                        <h4>📂 Detected Category</h4>
                        <div style="background: #667eea; color: white; padding: 10px 20px; border-radius: 25px; display: inline-block; margin-top: 10px;">
                            {category}
                        </div>
                    </div>
                    
                    <div style="margin: 20px 0;">
                        <h4>🎯 Identified Skills</h4>
                        <div style="margin-top: 10px;">
                            {skill_tags}
                        </div>
                    </div>
                </div>
                
                <h2>💼 Top Job Recommendations</h2>
                {job_cards}
                
                <div style="background: #e3f2fd; border-radius: 15px; padding: 25px; margin-top: 40px; border-left: 5px solid #2196F3;">
                    <h3 style="color: #1976d2; margin-bottom: 15px;">🚀 Career Enhancement Tips</h3>
                    <div style="background: white; padding: 15px; border-radius: 10px; margin: 10px 0;">
                        <strong>💡 Skill Development</strong>
                        <p>Focus on developing the most frequently required skills in your target jobs to increase your match scores.</p>
                    </div>
                    <div style="background: white; padding: 15px; border-radius: 10px; margin: 10px 0;">
                        <strong>📈 Resume Optimization</strong>
                        <p>Use action verbs, quantify achievements, and include relevant keywords from job descriptions.</p>
                    </div>
                    <div style="background: white; padding: 15px; border-radius: 10px; margin: 10px 0;">
                        <strong>🎯 Targeted Applications</strong>
                        <p>Tailor your resume for each application, emphasizing skills that match the job requirements.</p>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True, port=5006)