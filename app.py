from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import requests
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter

app = Flask(__name__)
app.config["SECRET_KEY"] = "resume-matcher-key"
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Initialize ML model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded successfully")


def extract_text_from_pdf(pdf_path):
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


def extract_skills(text):
    skills_db = [
        # Programming Languages
        "python",
        "java",
        "javascript",
        "typescript",
        "c++",
        "c#",
        "php",
        "ruby",
        "go",
        "rust",
        "swift",
        "kotlin",
        "scala",
        "r",
        "matlab",
        "perl",
        "shell",
        "bash",
        "powershell",
        # Web Technologies
        "react",
        "angular",
        "vue",
        "node.js",
        "django",
        "flask",
        "spring",
        "express",
        "html",
        "css",
        "sass",
        "bootstrap",
        "jquery",
        "webpack",
        "less",
        "tailwind",
        "laravel",
        # Databases
        "sql",
        "mysql",
        "postgresql",
        "mongodb",
        "redis",
        "oracle",
        "sqlite",
        "cassandra",
        "elasticsearch",
        "dynamodb",
        "firebase",
        # Cloud & DevOps
        "aws",
        "azure",
        "gcp",
        "docker",
        "kubernetes",
        "jenkins",
        "terraform",
        "ansible",
        "linux",
        "unix",
        "ci/cd",
        "gitlab",
        "github",
        "chef",
        "puppet",
        "vagrant",
        # Data & ML
        "machine learning",
        "data science",
        "tensorflow",
        "pytorch",
        "pandas",
        "numpy",
        "scikit-learn",
        "tableau",
        "power bi",
        "deep learning",
        "data analysis",
        "statistics",
        "excel",
        "spark",
        "hadoop",
        # Sales & Marketing
        "sales",
        "lead generation",
        "cold calling",
        "crm",
        "salesforce",
        "hubspot",
        "b2b sales",
        "b2c sales",
        "account management",
        "business development",
        "negotiation",
        "closing deals",
        "pipeline management",
        "customer relationship management",
        "prospecting",
        "sales forecasting",
        "territory management",
        "retail sales",
        "inside sales",
        "outside sales",
        "channel sales",
        "partnership development",
        "marketing",
        "digital marketing",
        "content marketing",
        "social media marketing",
        "email marketing",
        "seo",
        "sem",
        "ppc",
        "google ads",
        "facebook ads",
        "linkedin ads",
        "marketing automation",
        "brand management",
        "campaign management",
        "market research",
        "competitive analysis",
        # Digital Media & Creative
        "digital media",
        "social media",
        "content creation",
        "copywriting",
        "video editing",
        "photography",
        "graphic design",
        "adobe creative suite",
        "photoshop",
        "illustrator",
        "indesign",
        "premiere pro",
        "after effects",
        "figma",
        "sketch",
        "canva",
        "video production",
        "audio editing",
        "podcasting",
        "youtube",
        "instagram",
        "tiktok",
        "twitter",
        "facebook",
        "linkedin",
        "pinterest",
        "snapchat",
        "influencer marketing",
        "community management",
        "brand storytelling",
        "visual communication",
        # Culinary & Food Service
        "cooking",
        "culinary arts",
        "food preparation",
        "menu planning",
        "recipe development",
        "baking",
        "pastry",
        "food safety",
        "haccp",
        "kitchen management",
        "inventory management",
        "cost control",
        "food presentation",
        "plating",
        "wine pairing",
        "beverage service",
        "catering",
        "restaurant management",
        "food styling",
        "nutrition",
        "dietary restrictions",
        "allergen management",
        "sous vide",
        "grilling",
        "knife skills",
        "food costing",
        "vendor management",
        "staff training",
        "customer service",
        # Consulting & Business
        "consulting",
        "business analysis",
        "strategy",
        "process improvement",
        "change management",
        "project management",
        "stakeholder management",
        "requirements gathering",
        "problem solving",
        "analytical thinking",
        "presentation skills",
        "client management",
        "business case development",
        "roi analysis",
        "kpi development",
        "performance metrics",
        "operational excellence",
        "lean methodology",
        "six sigma",
        "business intelligence",
        "market analysis",
        "feasibility studies",
        # Arts & Creative
        "fine arts",
        "painting",
        "drawing",
        "sculpture",
        "ceramics",
        "printmaking",
        "mixed media",
        "art history",
        "art criticism",
        "gallery management",
        "exhibition planning",
        "art curation",
        "art education",
        "art therapy",
        "creative writing",
        "poetry",
        "screenwriting",
        "storytelling",
        "theater",
        "acting",
        "directing",
        "stage management",
        "costume design",
        "set design",
        "music",
        "composition",
        "performance",
        "music production",
        "sound engineering",
        # Automotive
        "automotive",
        "mechanical repair",
        "engine diagnostics",
        "brake systems",
        "transmission repair",
        "electrical systems",
        "air conditioning",
        "suspension",
        "alignment",
        "tire service",
        "automotive sales",
        "parts management",
        "service advisor",
        "warranty claims",
        "obd diagnostics",
        "hybrid vehicles",
        "electric vehicles",
        "automotive technology",
        "collision repair",
        "painting",
        "body work",
        "frame repair",
        "insurance claims",
        "automotive finance",
        # Design
        "ui design",
        "ux design",
        "user experience",
        "user interface",
        "wireframing",
        "prototyping",
        "design thinking",
        "user research",
        "usability testing",
        "information architecture",
        "interaction design",
        "visual design",
        "typography",
        "color theory",
        "layout design",
        "responsive design",
        "mobile design",
        "web design",
        "print design",
        "branding",
        "logo design",
        "packaging design",
        "industrial design",
        "product design",
        "interior design",
        "architectural design",
        "cad",
        "autocad",
        "3d modeling",
        "rendering",
        # Fitness & Health
        "personal training",
        "fitness coaching",
        "group fitness",
        "yoga instruction",
        "pilates",
        "strength training",
        "cardio training",
        "functional training",
        "sports conditioning",
        "injury prevention",
        "rehabilitation",
        "nutrition coaching",
        "meal planning",
        "weight management",
        "fitness assessment",
        "program design",
        "client motivation",
        "health education",
        "cpr certification",
        "first aid",
        "anatomy",
        "physiology",
        "exercise science",
        "sports medicine",
        "physical therapy",
        "massage therapy",
        "wellness coaching",
        # Finance & Accounting
        "accounting",
        "bookkeeping",
        "financial analysis",
        "budgeting",
        "forecasting",
        "tax preparation",
        "audit",
        "compliance",
        "financial reporting",
        "accounts payable",
        "accounts receivable",
        "payroll",
        "cost accounting",
        "management accounting",
        "financial planning",
        "investment analysis",
        "risk management",
        "insurance",
        "banking",
        "credit analysis",
        "loan processing",
        "financial modeling",
        "valuation",
        "mergers and acquisitions",
        "corporate finance",
        "quickbooks",
        "sage",
        "financial software",
        "gaap",
        "ifrs",
        "sox compliance",
        # General Business Skills
        "leadership",
        "team management",
        "communication",
        "public speaking",
        "presentation",
        "customer service",
        "problem solving",
        "critical thinking",
        "time management",
        "organization",
        "multitasking",
        "attention to detail",
        "adaptability",
        "teamwork",
        "collaboration",
        "conflict resolution",
        "decision making",
        "strategic thinking",
        "innovation",
        "creativity",
        "finance",
        "management",
        "powerpoint",
        "analytics",
        "agile",
        "scrum",
        # Other Technical
        "git",
        "jira",
        "confluence",
        "api",
        "rest",
        "graphql",
        "microservices",
        "testing",
        "debugging",
        "networking",
        "security",
        "blockchain",
        "cryptocurrency",
    ]

    text_lower = text.lower()
    found_skills = []

    for skill in skills_db:
        if skill in text_lower:
            found_skills.append(skill)

    # Also check for common variations
    skill_variations = {
        "js": "javascript",
        "ts": "typescript",
        "ml": "machine learning",
        "ai": "artificial intelligence",
        "ui/ux": "ui design",
        "frontend": "front-end development",
        "backend": "back-end development",
        "fullstack": "full-stack development",
    }

    for variation, skill in skill_variations.items():
        if variation in text_lower and skill not in found_skills:
            found_skills.append(skill)

    return list(set(found_skills))


def extract_experience(text):
    patterns = [
        r"(\d+)\+?\s*years?\s*of\s*experience",
        r"experience\s*[:\-]?\s*(\d+)\+?\s*years?",
        r"(\d+)\+?\s*years?\s*experience",
    ]

    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return int(match.group(1))
    return 0


def fetch_jobs_from_api(search=None, location=None):
    """
    Fetch jobs from multiple sources with proper fallback handling.
    """
    
    # ========== TRY ADZUNA API FIRST ==========
    app_id = "8b08337d"
    app_key = "732adbab8c05ccb0425581241a832428"
    url = "https://api.adzuna.com/v1/api/jobs/in/search/1"
    
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "results_per_page": 20,
        "content-type": "application/json"
    }
    
    if search:
        params["what"] = search
    if location:
        params["where"] = location
    
    try:
        print(f"Fetching jobs from Adzuna API...")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = data.get("results", [])
        
        if results:
            jobs = []
            for job in results:
                # Extract description safely
                description = job.get("description", "")
                if not description:
                    description = f"{job.get('title', 'Job')} position at {job.get('company', {}).get('display_name', 'company')}"
                
                jobs.append({
                    "title": job.get("title", "N/A"),
                    "company": job.get("company", {}).get("display_name", "N/A"),
                    "location": job.get("location", {}).get("display_name", "N/A"),
                    "description": description,
                    "salary": f"₹{job.get('salary_min', 0):,.0f} - ₹{job.get('salary_max', 0):,.0f}"
                              if job.get("salary_min") else "Not specified",
                    "url": job.get("redirect_url", "#"),
                    "contract_time": job.get("contract_time", "N/A")
                })
            
            print(f"✓ Successfully fetched {len(jobs)} jobs from Adzuna")
            return jobs
        else:
            print("No jobs found in Adzuna, trying Remotive...")
    
    except Exception as e:
        print(f"✗ Adzuna API Error: {e}")
    
    
    # ========== TRY REMOTIVE API AS FALLBACK ==========
    try:
        print("Fetching jobs from Remotive API...")
        remotive_url = "https://remotive.io/api/remote-jobs"
        remotive_params = {"limit": 20}
        
        response = requests.get(remotive_url, params=remotive_params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            remotive_jobs = data.get("jobs", [])
            
            if remotive_jobs:
                jobs = []
                for job in remotive_jobs[:20]:
                    description = job.get("description", "")
                    if not description:
                        description = f"Remote {job.get('title', 'position')} at {job.get('company_name', 'company')}"
                    
                    jobs.append({
                        "title": job.get("title", "N/A"),
                        "company": job.get("company_name", "N/A"),
                        "location": "Remote",
                        "description": description,
                        "salary": job.get("salary", "Not specified"),
                        "url": job.get("url", "#"),
                        "contract_time": "Remote"
                    })
                
                print(f"✓ Successfully fetched {len(jobs)} jobs from Remotive")
                return jobs
    
    except Exception as e:
        print(f"✗ Remotive API Error: {e}")
    
    
    # ========== FALLBACK TO HARDCODED JOBS ==========
    print("⚠ Using fallback job listings")
    return [
        {
            "title": "Senior Software Engineer",
            "company": "Google",
            "location": "Mountain View, CA",
            "description": "Design and develop large-scale software systems. Work with cutting-edge technologies including Python, Go, and distributed systems. Collaborate with cross-functional teams to deliver high-quality products.",
            "salary": "$150,000 - $250,000",
            "url": "https://careers.google.com/jobs/",
            "contract_time": "Full-time"
        },
        {
            "title": "Data Scientist",
            "company": "Microsoft",
            "location": "Seattle, WA",
            "description": "Apply machine learning and statistical analysis to solve complex business problems. Work with large datasets, build predictive models, and provide actionable insights to drive business decisions.",
            "salary": "$130,000 - $200,000",
            "url": "https://careers.microsoft.com/us/en",
            "contract_time": "Full-time"
        },
        {
            "title": "Product Marketing Manager",
            "company": "Apple",
            "location": "Cupertino, CA",
            "description": "Lead go-to-market strategies for innovative products. Develop marketing campaigns, conduct market research, and work closely with product teams to drive user adoption and engagement.",
            "salary": "$140,000 - $220,000",
            "url": "https://jobs.apple.com/",
            "contract_time": "Full-time"
        },
        {
            "title": "Senior Financial Analyst",
            "company": "Amazon",
            "location": "New York, NY",
            "description": "Perform financial analysis, budgeting, and forecasting. Support strategic decision-making through data-driven insights and financial modeling. Work with cross-functional teams on business planning.",
            "salary": "$100,000 - $150,000",
            "url": "https://www.amazon.jobs/",
            "contract_time": "Full-time"
        },
        {
            "title": "Sales Director",
            "company": "Salesforce",
            "location": "San Francisco, CA",
            "description": "Lead enterprise sales team to drive revenue growth. Develop strategic partnerships, manage key client relationships, and implement sales processes to achieve targets.",
            "salary": "$180,000 - $300,000",
            "url": "https://salesforce.wd1.myworkdayjobs.com/External_Career_Site",
            "contract_time": "Full-time"
        },
        {
            "title": "DevOps Engineer",
            "company": "Netflix",
            "location": "Los Gatos, CA",
            "description": "Build and maintain cloud infrastructure at scale. Implement CI/CD pipelines, monitor system performance, and ensure high availability of streaming services for millions of users.",
            "salary": "$160,000 - $240,000",
            "url": "https://jobs.netflix.com/",
            "contract_time": "Full-time"
        },
        {
            "title": "UX Designer",
            "company": "Meta",
            "location": "Menlo Park, CA",
            "description": "Design intuitive user experiences for social media platforms. Conduct user research, create wireframes and prototypes, and collaborate with product teams to enhance user engagement.",
            "salary": "$120,000 - $180,000",
            "url": "https://www.metacareers.com/",
            "contract_time": "Full-time"
        },
        {
            "title": "Machine Learning Engineer",
            "company": "Tesla",
            "location": "Palo Alto, CA",
            "description": "Develop AI systems for autonomous vehicles. Work on computer vision, deep learning models, and real-time inference systems to advance self-driving technology.",
            "salary": "$170,000 - $280,000",
            "url": "https://www.tesla.com/careers",
            "contract_time": "Full-time"
        },
    ]

def calculate_match_score(resume_text, resume_skills, job_description, job_title):
    # Generate embeddings using sentence transformer
    resume_embedding = model.encode(resume_text)
    job_embedding = model.encode(f"{job_title} {job_description}")

    # Calculate semantic similarity
    semantic_similarity = cosine_similarity(
        resume_embedding.reshape(1, -1), job_embedding.reshape(1, -1)
    )[0][0]

    # Extract skills from job description
    job_skills = extract_skills(job_description.lower() + " " + job_title.lower())

    # Calculate skill overlap with boost
    matching_skills = set(resume_skills) & set(job_skills)
    skill_score = len(matching_skills) / len(job_skills) if job_skills else 0

    # Boost scores for better user experience
    # Add base score to prevent very low scores
    base_score = 0.3
    boosted_semantic = min(1.0, semantic_similarity + base_score)
    boosted_skill = min(1.0, skill_score + 0.2) if skill_score > 0 else 0.4

    # Combined score with higher weights for better matching
    final_score = (boosted_semantic * 0.5) + (boosted_skill * 0.5)

    # Ensure minimum score of 40% for reasonable matches
    final_score = (
        max(final_score, 0.4)
        if matching_skills or semantic_similarity > 0.3
        else final_score
    )

    return {
        "score": round(
            min(final_score * 100, 95), 2
        ),  # Cap at 95% with 3 decimal precision
        "semantic_similarity": round(boosted_semantic * 100, 2),
        "skill_match": round(boosted_skill * 100, 2),
        "matching_skills": list(matching_skills),
        "job_skills": job_skills,
    }


@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Resume Matcher</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: Arial, sans-serif; background-color: #1a1a1a; color: #ffffff; }
            .container { max-width: 800px; margin: 0 auto; padding: 40px 20px; }
            .header { text-align: center; margin-bottom: 40px; }
            .header h1 { font-size: 2.5rem; margin-bottom: 10px; }
            .header p { color: #cccccc; font-size: 1.1rem; }
            .upload-form { background-color: #2a2a2a; padding: 40px; border-radius: 8px; text-align: center; }
            .file-input { margin: 20px 0; padding: 15px; background-color: #3a3a3a; border: 1px solid #555; color: #fff; width: 100%; max-width: 400px; }
            .submit-btn { background-color: #007bff; color: white; padding: 15px 30px; border: none; border-radius: 4px; font-size: 1.1rem; cursor: pointer; margin-top: 20px; }
            .submit-btn:hover { background-color: #0056b3; }
            .tech-info { margin-top: 40px; background-color: #2a2a2a; padding: 30px; border-radius: 8px; }
            .tech-info h3 { margin-bottom: 15px; color: #007bff; }
            .tech-info p { line-height: 1.6; color: #cccccc; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Resume Matcher</h1>
                <p>AI-powered job matching using semantic embeddings</p>
            </div>
            
            <form class="upload-form" action="/upload" method="post" enctype="multipart/form-data">
                <h2>Upload Resume</h2>
                <input type="file" name="resume" accept=".pdf" required class="file-input">
                <br>
                <button type="submit" class="submit-btn">Analyze Resume</button>
            </form>
            
            
        </div>
    </body>
    </html>
    """


@app.route("/upload", methods=["POST"])
@app.route("/upload", methods=["POST"])
def upload_file():
    # Check if file is in request
    if "resume" not in request.files:
        print("❌ Error: No file in request")
        return redirect(url_for("index"))

    file = request.files["resume"]
    
    # Check if filename is empty
    if file.filename == "":
        print("❌ Error: Empty filename")
        return redirect(url_for("index"))

    # Check if file is PDF
    if not (file and file.filename.lower().endswith(".pdf")):
        print("❌ Error: File is not a PDF")
        return redirect(url_for("index"))

    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        print(f"✓ File saved: {filepath}")

        # Extract text from PDF
        print("📄 Extracting text from PDF...")
        resume_text = extract_text_from_pdf(filepath)
        
        if not resume_text or len(resume_text.strip()) < 50:
            print(f"❌ Error: Insufficient text extracted (length: {len(resume_text)})")
            os.remove(filepath)
            return redirect(url_for("index"))
        
        print(f"✓ Extracted {len(resume_text)} characters from resume")

        # Extract skills and experience
        print("🔍 Extracting skills...")
        resume_skills = extract_skills(resume_text)
        print(f"✓ Found {len(resume_skills)} skills: {resume_skills[:5]}...")
        
        experience_years = extract_experience(resume_text)
        print(f"✓ Experience: {experience_years} years")

        # Fetch jobs from API
        print("🌐 Fetching jobs from API...")
        jobs = fetch_jobs_from_api()
        print(f"✓ Retrieved {len(jobs)} jobs")

        # Calculate matches
        print("🤖 Calculating job matches...")
        job_matches = []
        for i, job in enumerate(jobs, 1):
            try:
                match_data = calculate_match_score(
                    resume_text, 
                    resume_skills, 
                    job.get("description", ""), 
                    job.get("title", "")
                )
                job_matches.append({"job": job, "match_data": match_data})
                print(f"  Job {i}/{len(jobs)}: {job['title']} - Score: {match_data['score']}%")
            except Exception as e:
                print(f"  ⚠ Error matching job {i}: {e}")
                continue

        if not job_matches:
            print("❌ Error: No job matches calculated")
            os.remove(filepath)
            return redirect(url_for("index"))

        # Sort by score
        job_matches.sort(key=lambda x: x["match_data"]["score"], reverse=True)
        print(f"✓ Top match: {job_matches[0]['job']['title']} ({job_matches[0]['match_data']['score']}%)")

        # Clean up
        os.remove(filepath)
        print("✓ Cleaned up temporary file")

        # Generate results
        print("📊 Generating results page...")
        return generate_results_html(resume_skills, experience_years, job_matches)

    except Exception as e:
        print(f"❌ Fatal Error in upload_file: {e}")
        import traceback
        traceback.print_exc()
        
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        
        return redirect(url_for("index"))


def generate_results_html(resume_skills, experience_years, job_matches):
    skills_html = ", ".join(resume_skills) if resume_skills else "None detected"

    jobs_html = ""
    for i, match in enumerate(job_matches[:10], 1):
        job = match["job"]
        match_data = match["match_data"]

        matching_skills_html = (
            ", ".join(match_data["matching_skills"])
            if match_data["matching_skills"]
            else "None"
        )

        score = f"{match_data['score']:.2f}"
        semantic_sim = f"{match_data['semantic_similarity']:.2f}"
        skill_match = f"{match_data['skill_match']:.2f}"

        jobs_html += f'''
        <div class="job-card">
            <div class="job-header">
                <h3>{job["title"]}</h3>
                <div class="score">{score}%</div>
            </div>
            <div class="job-info">
                <p><strong>Company:</strong> {job["company"]}</p>
                <p><strong>Location:</strong> {job["location"]}</p>
                <p><strong>Salary:</strong> {job["salary"]}</p>
            </div>
            <div class="job-description">
                <p>{job["description"][:300]}...</p>
            </div>
            <div class="match-details">
                <p><strong>Semantic Similarity:</strong> {semantic_sim}%</p>
                <p><strong>Skill Match:</strong> {skill_match}%</p>
                <p><strong>Matching Skills:</strong> {matching_skills_html}</p>
            </div>
            <a href="{job["url"]}" target="_blank" class="apply-btn">Apply Now</a>
        </div>
        '''

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Resume Analysis Results</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: Arial, sans-serif; background-color: #1a1a1a; color: #ffffff; }}
            .container {{ max-width: 1000px; margin: 0 auto; padding: 40px 20px; }}
            .header {{ text-align: center; margin-bottom: 40px; }}
            .back-link {{ display: inline-block; background-color: #3a3a3a; color: #fff; padding: 10px 20px; text-decoration: none; margin-bottom: 30px; }}
            .summary {{ background-color: #2a2a2a; padding: 30px; border-radius: 8px; margin-bottom: 40px; }}
            .summary h2 {{ margin-bottom: 20px; }}
            .job-card {{ background-color: #2a2a2a; padding: 25px; margin-bottom: 20px; border-radius: 8px; border-left: 4px solid #007bff; }}
            .job-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }}
            .job-header h3 {{ color: #007bff; }}
            .score {{ font-size: 1.5rem; font-weight: bold; color: #28a745; }}
            .job-info {{ margin-bottom: 15px; }}
            .job-info p {{ margin-bottom: 5px; }}
            .job-description {{ margin-bottom: 15px; color: #cccccc; }}
            .match-details {{ margin-bottom: 15px; font-size: 0.9rem; }}
            .match-details p {{ margin-bottom: 3px; }}
            .apply-btn {{ background-color: #28a745; color: white; padding: 8px 16px; text-decoration: none; border-radius: 4px; }}
            .apply-btn:hover {{ background-color: #218838; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Resume Analysis Results</h1>
            </div>
            
            <a href="/" class="back-link">Upload Another Resume</a>
            
            <div class="summary">
                <h2>Resume Summary</h2>
                <p><strong>Skills Found:</strong> {skills_html}</p>
                <p><strong>Experience:</strong> {experience_years} years</p>
                <p><strong>Total Skills:</strong> {len(resume_skills)}</p>
            </div>
            
            <h2>Job Matches</h2>
            {jobs_html}
        </div>
    </body>
    </html>
    """


if __name__ == "__main__":
    import os
    port = int(os.environ.get('PORT', 5009))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
