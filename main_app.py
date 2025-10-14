from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import re
from collections import Counter

app = Flask(__name__)
app.config['SECRET_KEY'] = 'resume-matcher-secret-key'
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
        'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin', 'go', 'programming', 'coding', 'software development'],
        'web': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'web development', 'frontend', 'backend', 'full stack'],
        'data': ['sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'database', 'data analysis', 'excel', 'tableau', 'power bi'],
        'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'cloud computing', 'devops', 'terraform'],
        'mobile': ['android', 'ios', 'mobile development', 'react native', 'flutter'],
        
        # FINANCE & ACCOUNTING
        'finance': ['financial analysis', 'investment', 'portfolio management', 'budgeting', 'forecasting', 'finance', 'financial modeling'],
        'accounting': ['accounting', 'bookkeeping', 'tax preparation', 'auditing', 'financial reporting', 'gaap', 'quickbooks', 'sap'],
        'banking': ['banking', 'loans', 'credit analysis', 'relationship management', 'financial products', 'risk management'],
        
        # LEGAL
        'legal': ['legal research', 'contract law', 'corporate law', 'litigation', 'legal writing', 'law', 'attorney', 'lawyer', 'compliance'],
        
        # HEALTHCARE
        'healthcare': ['patient care', 'nursing', 'medical procedures', 'clinical skills', 'healthcare', 'medical', 'healthcare management'],
        'medical': ['diagnosis', 'treatment', 'medical records', 'pharmacology', 'anatomy', 'physiology'],
        
        # EDUCATION
        'education': ['teaching', 'curriculum development', 'lesson planning', 'classroom management', 'education', 'training'],
        'academic': ['research', 'academic writing', 'educational technology', 'instructional design'],
        
        # SALES & MARKETING
        'sales': ['sales management', 'lead generation', 'negotiation', 'crm', 'sales', 'business development', 'account management'],
        'marketing': ['digital marketing', 'social media', 'seo', 'content marketing', 'marketing', 'advertising', 'brand management'],
        'communications': ['public relations', 'media relations', 'communications', 'content creation', 'copywriting'],
        
        # HUMAN RESOURCES
        'hr': ['human resources', 'recruiting', 'talent acquisition', 'employee relations', 'hr', 'performance management'],
        'recruitment': ['sourcing', 'interviewing', 'candidate assessment', 'onboarding', 'talent management'],
        
        # ENGINEERING
        'mechanical': ['mechanical engineering', 'cad', 'solidworks', 'autocad', 'manufacturing', 'design'],
        'civil': ['civil engineering', 'structural design', 'construction management', 'project planning'],
        'electrical': ['electrical engineering', 'circuit design', 'power systems', 'electronics'],
        
        # DESIGN & CREATIVE
        'design': ['graphic design', 'visual design', 'branding', 'typography', 'creative direction'],
        'ux_ui': ['ux design', 'ui design', 'user experience', 'prototyping', 'wireframing', 'figma'],
        'creative': ['creativity', 'artistic', 'illustration', 'photography', 'video editing'],
        
        # OTHER INDUSTRIES
        'culinary': ['culinary arts', 'cooking', 'food preparation', 'menu planning', 'chef', 'kitchen management'],
        'construction': ['construction management', 'project management', 'safety management', 'construction'],
        'agriculture': ['farming', 'crop management', 'agriculture', 'agricultural', 'sustainability'],
        'fitness': ['personal training', 'fitness coaching', 'nutrition', 'fitness', 'wellness'],
        'consulting': ['business consulting', 'strategy consulting', 'consulting', 'advisory'],
        'aviation': ['flight operations', 'aviation', 'pilot', 'aircraft maintenance'],
        
        # SOFT SKILLS
        'leadership': ['leadership', 'team management', 'mentoring', 'management', 'supervision'],
        'communication': ['communication', 'presentation', 'public speaking', 'interpersonal'],
        'analytical': ['analytical thinking', 'problem solving', 'research', 'analysis', 'critical thinking'],
        'organizational': ['organization', 'time management', 'multitasking', 'planning', 'coordination']
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
        r'(\d+)\+?\s*years?\s*experience',
        r'(\d+)\+?\s*yrs?\s*experience',
        r'experience\s*[:\-]?\s*(\d+)\+?\s*yrs?'
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
        'Digital Media': ['digital media', 'content creator', 'social media manager', 'digital marketing']
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
    {'title': 'DevOps Engineer', 'company': 'Cloud Systems', 'skills': ['aws', 'kubernetes', 'docker', 'jenkins', 'terraform', 'linux'], 'salary': '$110k-$150k', 'category': 'Information Technology', 'description': 'Manage cloud infrastructure and CI/CD pipelines.'},
    {'title': 'Data Analyst', 'company': 'Analytics Pro', 'skills': ['sql', 'python', 'excel', 'tableau', 'data analysis'], 'salary': '$70k-$100k', 'category': 'Information Technology', 'description': 'Analyze data and create business intelligence reports.'},
    
    # FINANCE & ACCOUNTING
    {'title': 'Senior Accountant', 'company': 'Financial Services LLC', 'skills': ['accounting', 'financial reporting', 'tax preparation', 'excel', 'gaap'], 'salary': '$65k-$85k', 'category': 'Accountant', 'description': 'Handle complex accounting tasks, financial reporting, and tax compliance.'},
    {'title': 'Financial Analyst', 'company': 'Investment Group', 'skills': ['financial analysis', 'excel', 'investment', 'budgeting', 'finance'], 'salary': '$70k-$95k', 'category': 'Finance', 'description': 'Analyze financial data and provide investment recommendations.'},
    {'title': 'Banking Relationship Manager', 'company': 'First National Bank', 'skills': ['banking', 'relationship management', 'sales', 'financial products'], 'salary': '$60k-$80k', 'category': 'Banking', 'description': 'Manage client relationships and develop banking solutions.'},
    {'title': 'Tax Specialist', 'company': 'Tax Advisory Group', 'skills': ['tax preparation', 'accounting', 'compliance', 'financial reporting'], 'salary': '$55k-$75k', 'category': 'Accountant', 'description': 'Prepare tax returns and provide tax planning advice.'},
    
    # LEGAL
    {'title': 'Corporate Lawyer', 'company': 'Legal Associates', 'skills': ['legal research', 'contract law', 'corporate law', 'litigation'], 'salary': '$120k-$180k', 'category': 'Advocate', 'description': 'Handle corporate legal matters and contract negotiations.'},
    {'title': 'Legal Assistant', 'company': 'Law Firm Partners', 'skills': ['legal research', 'legal writing', 'case management'], 'salary': '$45k-$60k', 'category': 'Advocate', 'description': 'Support attorneys with research and document preparation.'},
    {'title': 'Compliance Officer', 'company': 'Regulatory Solutions', 'skills': ['compliance', 'legal research', 'risk management', 'regulatory'], 'salary': '$80k-$110k', 'category': 'Advocate', 'description': 'Ensure organizational compliance with laws and regulations.'},
    
    # HEALTHCARE
    {'title': 'Registered Nurse', 'company': 'City Medical Center', 'skills': ['patient care', 'nursing', 'medical procedures', 'healthcare'], 'salary': '$65k-$85k', 'category': 'Healthcare', 'description': 'Provide comprehensive patient care and medical support.'},
    {'title': 'Healthcare Administrator', 'company': 'Regional Hospital', 'skills': ['healthcare', 'management', 'budgeting', 'leadership'], 'salary': '$75k-$105k', 'category': 'Healthcare', 'description': 'Manage healthcare operations and administrative functions.'},
    {'title': 'Medical Assistant', 'company': 'Family Practice', 'skills': ['patient care', 'medical procedures', 'healthcare', 'medical records'], 'salary': '$35k-$45k', 'category': 'Healthcare', 'description': 'Assist physicians with patient care and administrative tasks.'},
    
    # EDUCATION
    {'title': 'High School Mathematics Teacher', 'company': 'Lincoln High School', 'skills': ['teaching', 'education', 'curriculum development', 'classroom management'], 'salary': '$45k-$65k', 'category': 'Teacher', 'description': 'Teach mathematics and develop engaging lesson plans.'},
    {'title': 'Elementary School Teacher', 'company': 'Sunshine Elementary', 'skills': ['teaching', 'education', 'classroom management', 'communication'], 'salary': '$42k-$62k', 'category': 'Teacher', 'description': 'Educate elementary students across multiple subjects.'},
    {'title': 'Training Specialist', 'company': 'Corporate Learning', 'skills': ['training', 'education', 'curriculum development', 'presentation'], 'salary': '$55k-$75k', 'category': 'Teacher', 'description': 'Develop and deliver corporate training programs.'},
    
    # SALES & MARKETING
    {'title': 'Sales Manager', 'company': 'Enterprise Solutions', 'skills': ['sales management', 'leadership', 'business development', 'crm'], 'salary': '$70k-$100k', 'category': 'Sales', 'description': 'Lead sales team and develop revenue growth strategies.'},
    {'title': 'Digital Marketing Specialist', 'company': 'Creative Agency', 'skills': ['digital marketing', 'social media', 'seo', 'content marketing'], 'salary': '$50k-$70k', 'category': 'Marketing', 'description': 'Create and execute digital marketing campaigns.'},
    {'title': 'Account Executive', 'company': 'Media Group', 'skills': ['sales', 'account management', 'communication', 'negotiation'], 'salary': '$55k-$80k', 'category': 'Sales', 'description': 'Manage client accounts and drive new business development.'},
    
    # HUMAN RESOURCES
    {'title': 'HR Business Partner', 'company': 'Global Corporation', 'skills': ['human resources', 'talent management', 'employee relations', 'leadership'], 'salary': '$80k-$110k', 'category': 'HR', 'description': 'Partner with business leaders on HR strategy and talent development.'},
    {'title': 'Recruiter', 'company': 'Talent Acquisition Inc', 'skills': ['recruiting', 'talent acquisition', 'hr', 'communication'], 'salary': '$55k-$75k', 'category': 'HR', 'description': 'Source and recruit top talent for various positions.'},
    {'title': 'HR Generalist', 'crt=5004)bug=True, po app.run(de  in__':
 _ == '__mae_

if __nam>
    '''/html    <
  </body></div>
  
        n>mai          </on>
  /secti <           div>
     </               
    nts.</p>b requiremech the jolls that matizing skiemphascation, each applie for umour resilor yp>Ta   <                h4>
     ations</Applich4>Targeted         <              m">
  "tip-ites=clasv <di            >
              </div         .</p>
     escriptionsjob drom t keywords fanlude relevd incnts, anvemefy achies, quantierbse action v      <p>U                ion</h4>
  me Optimizatesu>R<h4                       em">
 it"tip-div class=      <         
       </div>          p>
        scores.</r match crease youbs to in target joour yed skills inquiry ret frequentlthe moseveloping s on dp>Focu        <     
           </h4>opmentvelkill De       <h4>S         
        m">te"tip-iclass=iv    <d               </h3>
  ps Tientancemer Enhre>Ca     <h3         >
      "on"tips-secti class=ionect         <s         
          tion>
    ec        </s       
 job_cards}        {     2>
       </hmmendations>Job Reco<h2               ">
     jobs-sectionon class="cti   <se                  
      
     /section>      <
          </div>                   /div>
 ll_tags}<{ski <div>                  
     ills</h3>Sk>Identified  <h3                     >
  er"containills-s="skclas  <div          
                          >
         </div        n>
      /spary}<egoat-badge">{c"categoryass=    <span cl                </h3>
    egorytected Cat <h3>De                     v>
     <di             
                     div>
   </                    div>
        </           
     hed</span> Matc>Jobsary-label"mm="sulasspan c       <s                 pan>
    s)}</satchen(job_mnumber">{lesummary-ss=" <span cla                          ">
 ry-itemsummaiv class="    <d             >
       /div    <                   ed</span>
 s Analyz">Wordbel"summary-la= class      <span                   /span>
   ))}<ext.split(ber">{len(tary-num"summn class=       <spa               >
      tem""summary-iiv class=    <d                    
     </div>           >
        ence</spans Experilabel">Year="summary-lass c <span                           }</span>
else 'N/A'ce  experienience if{expermber">y-nuummar="sn class      <spa                ">
      summary-item="<div class                      </div>
                  >
        d</spans Identifieel">Skillabary-lsumm class="      <span                     /span>
 kills)}<r">{len(smbe"summary-nu class=<span                           item">
 mmary-"sudiv class=     <               id">
    "summary-griv class=        <d          2>
  </hummarye S>Resum  <h2                 ">
 aryummon class="scti      <se                  
 /a>
       ther Resume< AnoUpload">← back-link=" class/"f="re        <a h     >
   "content"main class=  <                
er>
            </headp>
      dations</commenres and job e analysiivehensmpr       <p>Co         </h1>
Resultss ysiume Anal>Res      <h1        ">
  erass="headcler ad         <heer">
   ainass="contv cl   <di     ody>
ad>
    <b
    </he  </style>        }}
       
      }}          enter;
   ntent: ccotify-        jus            down {{
-break     .score         
                  
       }}
         umns: 1fr;e-coltemplat  grid-                 ion {{
 lls-sectski          .     
             }}
                    16px;
   gap:             ;
     n: columnlex-directio           f         {{
 derjob-hea         .  
                  }}
              0px;
     g: 2addin     p            {{
    ontent  .c              768px) {{
x-width: ma@media (            
            }}
            : 1.5;
ine-height   l        49e;
     lor: #8b9      co
           p {{itemp-     .ti     
            }}
              ;
 1rem font-size:             x;
  -bottom: 8p  margin        
      ff;r: #58a6colo         {
       -item h4 {ip         .t  
          }
            }ff;
   a6px solid #58r-left: 4   borde       ;
      om: 16pxgin-bott    mar         0px;
    2    padding:            2;
olor: #161b2ackground-c   b   
          {.tip-item {    
                  }
         }5rem;
     t-size: 1.2fon               ;
 24px: rgin-bottom    ma            #f0f6fc;
     color:          
  ion h3 {{ectips-s.t                  
    }}
            px;
  p: 40-to margin            px;
   adding: 32           p
     id #30363d;px solr: 1 borde            
   7;r: #0d111loround-co      backg
          section {{ips-       .t  
     
                  }}ic;
    -style: ital      font
          b949e; color: #8        
       {{ills no-sk          .       
         }}
         0.8rem;
 size: ont-          f       2px;
argin:   m        
     4px 8px;ding:       pad
          fff;lor: #fff    co          da3633;
  und-color: #ckgro ba             e-block;
  lay: inlin     disp      g {{
     ll-missinki .s               
          }}
   em;
       : 0.8r font-size               ;
 margin: 2px          ;
     8px: 4px   padding             
 #ffffff;    color:             6;
#23863nd-color: ackgrou       b       ;
  lockinline-by:      displa      h {{
      .skill-matc         
          
         }}       12px;
 -bottom:ginmar                600;
 -weight:   font  
            0.9rem;ize:  font-s          c;
    f6fr: #f0olo  c           
    {{itle-tills.sk           
           
     }}      ;
   id #30363d: 1px sol   border      x;
       ing: 20p    padd          
  0d1117;olor: #kground-c      bac         group {{
 s-     .skill      
           }}
            ;
  24px gap:        r;
        : 1fr 1flate-columns   grid-temp           rid;
  : gplay     dis           ction {{
ills-se .sk            
          }}
            ;
 ight: 1.6    line-he       
     x;-bottom: 24p     margin          df3;
 or: #e6e       col         tion {{
ipscr     .job-de  
            }}
               op: 4px;
  argin-t  m         ;
      0.8remt-size:       fon  e;
        #8b949or:   col           abel {{
  onus-lxt, .b  .score-te                 
         }}
 
       r: #56d364;lo co               
nus-value {{bo          .       
  }
       }       ck;
   y: blo displa              
 8a6ff;  color: #5              00;
-weight: 6  font              m;
size: 1.25reont-       f    ue {{
     us-val.bonr, becore-num        .s   
        }}
                px;
 100idth:    min-w           363d;
  x solid #30order: 1p          b       16px;
ng:ddi        pa;
        or: #0d1117kground-colbac            
    enter;: cgnali    text-           {{
 .bonus-item re-item,      .sco    
                    }}
;
        wraprap:     flex-w          om: 24px;
  margin-bott            x;
   4pgap: 2               ex;
 display: fl           
      {{wndoe-break.scor          
            }}
           8rem;
   e: 0.iz font-s              b949e;
 or: #8    col          abel {{
  core-l  .s            
         }}
         ck;
    ay: blo  displ            
  d364;lor: #56        co
        ght: 700;-wei    font       
     size: 2rem;  font-              -value {{
   .score           
     }
             }
    80px;th: in-wid  m         nter;
     xt-align: ce te    
           {ch-score { .mat    
                 
   }}           9rem;
ize: 0.nt-s      fo     
     #8b949e;or: ol         c
       gory {{te         .ca   
   }
               }0;
      : 50 font-weight          364;
     or: #56dcol           {{
      lary       .sa  
              }
  }           t: 500;
 font-weigh            6ff;
   lor: #58a   co           {{
      .company          
          
 }}            p: wrap;
ex-wra     fl     6px;
       gap: 1             
  ex;isplay: fl         d       a {{
metob-     .j         
      
       }}      
   bottom: 8px;n-margi                 #f0f6fc;
  color:            
  t: 600; font-weigh            ;
   rem.25ze: 1nt-si    fo         le {{
   it     .job-t     
             }
  }           4px;
bottom: 2margin-              rt;
  x-stams: fle   align-ite           etween;
  t: space-btify-conten jus           lex;
     fy:la       disp{
          {-header.job            
    
          }}
          ;(-2px)lateYform: trans       trans    f;
      #58a6fder-color:or       b
         ard:hover {{ob-c   .j
                 }}
               .2s ease;
 l 0sition: al       tran  
       x;ttom: 24prgin-bo     ma        px;
   adding: 32    p          30363d;
  id # 1px sol  border:          b22;
    #161d-color: backgroun           
     ob-card {{        .j              
   }}
  
         m: 32px;n-bottogi       mar
         f0f6fc;r: #olo       c        
 rem;: 1.75ze   font-si            h2 {{
 tion jobs-sec  .       
             }}
         
     eft: 8px; margin-l              : italic;
 -style  font              #8b949e;
       color:        {{
  -more skill   .
                  }}
         00;
      ight: 5 font-we            
   e: 0.85rem;ont-siz      f
          2px; margin:              x 12px;
  dding: 4p  pa           ffff;
   #fflor:           co      : #1f6feb;
nd-colorgrouack     b     
      block;inline-:      display         {
  .skill-tag {               
    }}
             em;
    ize: 1.1rnt-s       fo;
         bottom: 16pxmargin-         
       ;fcf0f6r: # colo        {{
       3 r hls-containe     .skil   
               }}
          
   p: 24px;in-to      marg
          ontainer {{ills-c.sk                   
   }}
           x 0;
   rgin: 16p      ma
          t: 600;ont-weigh     f           16px;
 dding: 8px        pa        ffff;
#ff:       color        36;
  r: #2386round-colo      backg          ne-block;
play: inli     dis        
   ge {{category-bad .          
          }}
               em;
0.9rt-size:          fon9e;
       olor: #8b94           c{{
     -label summary  .
              }}
                8px;
     om:in-bottrgma           ;
     ock display: bl          
     f;lor: #58a6f    co         
    700;eight:    font-w         
   ;t-size: 2rem     fon           ber {{
-numryma  .sum
                      }}
            30363d;
d #x solider: 1p    bor           
 x;16p 24px ding:pad               17;
 11r: #0dround-colo  backg              enter;
ext-align: c t      
          {{emry-it.summa           
           }}
           
   om: 32px;tt   margin-bo        24px;
           gap:    );
        1fr)ax(150px,o-fit, minmutns: repeat(a-columtemplateid-gr            grid;
      display:             id {{
  mary-gr     .sum      
             }}
        px;
    bottom: 32n-     margi   
        c;#f0f6f     color:        ;
     1.5rem font-size:        
       ary h2 {{mm  .su       
      
            }}
         ;30363dsolid #: 1px    border             px;
om: 40n-bott      margi         px;
 dding: 40     pa        ;
   b22#161: d-color  backgroun         
     mmary {{       .su    
     
                 }}px);
   ranslateY(-1ransform: t    t   ;
         0363d#3olor: und-cro backg              {{
  :hoverback-link   .
                    }}
           
  ease; 0.2s nsition: all         tra      
 #30363d;solid 1px der: or  b              m: 40px;
rgin-botto   ma    
         ;neon: noorati  text-dec             ;
 12px 24pxing:         padd
        edf3;or: #e6 col           
    d;lor: #21262nd-co   backgrou         block;
    : inline-     display           -link {{
       .back     
                    }}
x;
    dding: 40p       pa
         ontent {{ .c     
                }}
      
        em; 1.1rt-size:         fone;
       olor: #8b949 c             {{
   r pde       .hea          
      }}
             om: 12px;
rgin-bott      ma      0f6fc;
    color: #f                00;
eight: 6    font-w            em;
size: 2.5r  font-       
       er h1 {{ead   .h                
        }}
;
         id #30363d: 1px solttomorder-bo  b          er;
    gn: centtext-ali              0px;
    padding: 4           61b22;
   : #1round-colorckg        ba   r {{
          .heade
                    }}
   
        ding: 0;    pad            n: 0 auto;
   margi       
      00px;dth: 12   max-wi        
     ainer {{   .cont 
               }}
             
     100vh;ight:min-he        
        t: 1.6;heighline-               edf3;
 #e6 color:             0d1117;
   -color: #ackground        b     ;
    sans-serif Cantarell,u,nten, Ubuoboto, Oxyg', Rt, 'Segoe UIMacSystemFonBlink, le-system: -appnt-family       fo         {{
       body      
               }}

         box;: border- box-sizing         
       0;g:      paddin     ;
      margin: 0                  * {{
  
        <style>    itle>
   /tults<is Res Analysesume<title>R">
        0cale=1.l-sth, initiavice-wididth=deent="wport" conte="viewa nam        <met"UTF-8">
set=<meta char        <head>
 ">
   g="enlan <html 
   tml>YPE h!DOCT   <''
 f'   return   
 ''
          'div>
</           </div>
   
      e ''} elskills']_ssingmismatch['>' if ></divs_html}</diving_skillsst">{mi-lislskilclass="siv 4><d])})</hs"issing_skillh["mlen(matc ({ to Developillsle">Sktits="skills-h4 clas-group"><skillslass="v c    {f'<di    
                  v>
            </di      /div>
    tml}<lls_hking_schit">{matlisills-skdiv class="    <                
})</h4>ills'])matching_sken(match[' Skills ({l>Matching-title"ss="skillscla        <h4           roup">
  s-glass="skilliv c         <d   ">
    onills-sectiss="skiv cla  <d           
       </p>
    iption']}escr{job['dcription">b-deslass="jo        <p c           
 </div>
         s}
       dicator   {bonus_in            iv>
          </dan>
       >Skills</spxt"score-tes="clas      <span     
          %</span>l_score']}{match['skilber">-num"scoreclass= <span             >
       m"ore-ite class="scdiv   <          n">
   ore-breakdow"scass=div cl       <  
              v>
   </di    iv>
      </d       
         pan>/s">Match<belore-la class="sc       <span          </span>
   e']}%ch['scorvalue">{matre-"scospan class=     <            >
   -score""matchlass=<div c       
             </div>      >
             </div           
  </span>al')}er'Genry', ego'catget(">{job.categoryan class="      <sp                 an>
 lary']}</spjob['salary">{class="sa  <span                     </span>
  ']}b['company">{joanymplass="cospan c           <            
 ta">b-mess="jo cla  <div               >
   ']}</h3leb['tit">{jotitleass="job-     <h3 cl               nfo">
="job-iclass     <div        ">
    -headerss="job  <div cla   
       job-card">lass="div c   <  = f'''
   ards + job_c   
       
     ></div>'an/spience Bonus<abel">Exper"bonus-lss=an clasp/span><s"]}%<"exp_bonu{match[">+value"bonus-class=span -item"><"bonusass=<div cl += f'atorsdicnus_in        bo0:
    nus'] > atch['exp_bof m  i  v>'
    /span></ditch<y Maegorbel">Cats="bonus-laclaspan span><sus"]}%</ategory_bon{match["c>+onus-value""bpan class="><s-itemnuss="bo f'<div clasors +=ats_indic bonu           nus'] > 0:
category_bo['tchf ma    i
    tors = ''ndica  bonus_i        
    :8]])
  kills'][['missing_s match inllor ski' fl}</span>il{sk-missing">ills="skclaspan .join([f'<s= ''l kills_htmmissing_s         
    /span>'
   nd< foulstching skil mas">Nos="no-skill'<span claslls_html = hing_ski       matcml:
     ng_skills_htt matchi       if nos']])
 killg_shintcmatch['man for skill i/span>' kill}<tch">{sill-maclass="skin([f'<span tml = ''.jong_skills_htchi ma
               'job']
ob = match[
        j1):, tches[:10]rate(job_match in enume  for i, ma = ''
  ardss
    job_cate job cardnerGe   
    # n>'
 pa0} more</s 2n(skills) -+{lel-more">s="skilpan clas += f'<s skill_tags      20:
  ) >lskil if len(s:20]])
    skills[or skill in}</span>' fskillkill-tag">{n class="sjoin([f'<spal_tags = ''.il
    skdisplay)r first 20 foimit to ags (lskill terate    # Gen""
    
 ML"results HTe the ""Generat
    "ches): job_matory, text,, categperience(skills, exults_htmlresate_f generde

index'))for('direct(url_turn re     re
    else:)
    
   or('index')url_ft(irecrn redretu            :
on as excepti   except E     
     
       ults_html return res  
         matches) job_gory, text,e, caterienc expes,illults_html(skrate_resnes_html = ge result          L
 HTM results   # Generate                 
   h)
  e(filepatremov         os. up
   an      # Cle 
               
  rue)everse=T rcore'],bda x: x['sy=lames.sort(ke job_match        re
    scort by # So                    
 })
              )
     set(skills)lls']) -ob['skit(js': list(sessing_skill     'mi      ,
         ills)atching_sk(m': listching_skillsat         'm           p_bonus,
us': ex   'exp_bon          us,
       tegory_bononus': ca'category_b                e, 1),
    ill_scorund(sk_score': ro  'skill               1),
   score, nal_ round(fi': 'score                   ': job,
    'job             ppend({
   _matches.a     job                   
    , 100)
    + exp_bonuss egory_bonure + catill_scoskin(= minal_score       f  re
        Final sco     #      
                     us = 5
 _bon       exp               1:
   rience >=   elif expe            
     0_bonus = 1       exp            >= 3:
     ce lif experien      e           5
   s = 1    exp_bonu            5:
         >= rienceif expe                 ence:
   f experi         i       bonus = 0
exp_             
   nce bonusrie     # Expe        
                  
 0tegory else = cay') =tegor job.get('ca5 ifry_bonus = 2goate     c     s
      nugory bo   # Cate            
        
         else 0b['skills']  jo* 100 if'skills']) / len(job[ng_skills) en(matchicore = l_skill       s       ])
  job['skills'ills) & set(s = set(skching_skillmat           OBS:
     for job in J         = []
   ches      job_matg
       anced scorin with enhhing jobsnd matc      # Fi          
        , skills)
extesume(tze_r= categoricategory             
ience(text)expertract_ce = exrien        expe   )
 texts(ensive_skillact_comprehls = extr      skil
               ex'))
   indurl_for('ct(rn redire        retu     t text:
    no   if   
      (filepath)_pdfomfrtext_ extract_ext =     te
       alyz and anxtract         # E
           th)
    filepaave( file.s        e)
   , filenamD_FOLDER']'UPLOA.config[ppath.join(aos.plepath = fi  
          e.filename)ilename(fil_fureename = secil        fry:
            t'.pdf'):
th(wiwer().endsfilename.loand file.le f fi   
    i
 'index'))t(url_for(edirec rreturn
        ':= 'lename =if file.fi
    e']sumfiles['reuest.ile = req 
    f
   ndex'))for('idirect(url_turn re      re  .files:
t in requestsume' nore):
    if 'e(_filploaddef uT'])
'POS', methods=[ute('/upload.ro'''

@app   >
  </htmly>
   </bod  </div>
  n>
        /mai   <    
     section>      </
          v> </di                   </p>
ss.or succeotential fnd ptibility aked by compaations ranb recommendnalized jorsoeceive pep>R     <              /h3>
     tions< RecommendaTargeted3>        <h        
        ">🎯</div>ture-icons="fea   <div clas            >
         "feature" class=div         <           v>
 </di                  ent.</p>
 egory alignmvel, and catnce lerie expey,patibilit comon skillsd s base match scoreprehensiveet com        <p>G                oring</h3>
ailed Sc3>Det    <h           >
         div>📊</n"ture-icoass="fea    <div cl                 
   "feature">v class=         <di           div>
   </                 >
n.</p precisiogh hiithtions wqualificarience, and lls, expeg skiintractontent, exe cur resumze yoithms analyanced algor <p>Adv                     h3>
  </lysisnt Anatellige   <h3>In                  v>
   ">🧠</dionure-icclass="feat       <div               ure">
   "feat class=      <div          es">
    turfeaclass="   <section               
           ion>
     </sect           
    </div>                   /span>
 ate<curacy Rl">Ac"stat-labean class=        <sp            >
    </spanber">95%um"stat-ns=<span clas                  ">
      tatv class="s     <di               /div>
    <       
         hing</span>Powered Matctat-label">"sclass=span          <             /span>
  ">AI<-numberass="stat<span cl                    
    tat"> class="s  <div             
           </div>              /span>
d<keSkills Tracabel">s="stat-lpan clas      <s               an>
   0+</spnumber">20"stat-pan class=   <s                    tat">
 ="s<div class                
    </div>                pan>
    /sies<ob Categor">Jelstat-lab class="   <span             >
        </span>30+-number"s="statspan clas         <        >
       "stat"s=div clas        <           "stats">
 ion class=ect       <s             
  
          /section> <            rm>
         </fo          >
    /div     <                on>
   e</buttze ResumAnalyy">btn-primarclass="mit" ype="subn tto   <but               
                 <br>                   nput">
  le-ifid class="equire".pdf" raccept=resume"  name="e="file"<input typ                  
          /p> to browse<r clickesume oyour PDF rp ro">Drag and ditlebtd-su"uploap class=   <                  
       2>ume</hResour >Upload Y"-title"uploadh2 class=        <                    >
divicon">📄</oad- class="upl       <div                     d-area">
loaupss=" <div cla                     data">
  form-ipart/ctype="multst" en"pothod=load" meup"/orm action=<f            
        ">d-sectionuploa"lass=n c <sectio              
 ntent">main-cos=" clas<main                 
  
     er>     </head>
        scoring</ps andlyticailed ana detwithmmendations cozed job renaliget persoesume and  your r<p>Upload       
         h1> Matcher</I Resume1>A    <h      >
      header"er class="    <head
        ntainer">co class="iv     <dody>
   ead>
    <b
    </hyle>   </st
            }   
   8px;argin-top:    m           0.9rem;
 ize: nt-s fo              e;
 lor: #8b949        co        el {
tat-lab        .s     
     }
            
      ay: block;       displ       6ff;
  or: #58a      col      700;
    t-weight: fon          ;
       2.5remize:      font-s        {
   stat-number      .       
     }
              ;
    center: gnt-ali tex       {
        .stat                     
     }
           63d;
#303d x soli: 1p      border
          : #161b22;nd-colorackgrou      b        
  px;: 40 padding              
 0px 0;margin: 8            x;
    2p    gap: 3           );
 px, 1fr)inmax(200it, meat(auto-fepmns: rmplate-colugrid-te            id;
    lay: gr      disp    {
       stats      .           
      }
   
          ight: 1.6;line-he       ;
         #8b949elor:      co      p {
         .feature        
                }
   
      ottom: 16px;argin-b           mfc;
     : #f0f6or     col          t: 600;
   font-weigh           25rem;
   ze: 1.t-si     fon           ature h3 {
     .fe   
                    }
        ;
ttom: 24pxmargin-bo           
     ff;#58a6color:         
        ;: 3rem   font-size           con {
   .feature-i              
            }
        0363d;
 #3solid x der: 1p       bor;
         er: centgnli    text-a       x;
     x 32p: 40padding    p       ;
      #161b22ound-color:   backgr   
          re {     .featu              
    }
       ;
       80pxargin-top:         m       40px;
:        gap          1fr));
x,(300p minmaxauto-fit,peat(olumns: remplate-cid-te       gr       rid;
  ay: g  displ             {
     .features     
           
               }
  Y(-1px);translatesform:   tran             43;
 color: #2ea0 background-      {
         ver mary:ho-pri      .btn   
               }
            p: 24px;
margin-to               s ease;
 on: all 0.2ransiti       t        pointer;
 r:  curso             
  ight: 600;font-we               1rem;
 1.nt-size:    fo        ;
     32px: 16px padding        ;
         noneer:  bord           
   ffff; #ffolor:         c
       36;r: #2386-colockground       ba
         imary {    .btn-pr
                    }
            px;
00: 4x-width   ma            
 100%;   width:              
 1rem;e:    font-siz           #e6edf3;
 : lor         co       
lid #30363d; so 1px    border:       62d;
     #212-color: background              x;
   16pg: 12px   paddin             ;
: 24px 0gin    mar      ut {
      e-inp       .fil              
         }
     2px;
 ttom: 3gin-bo    mar          49e;
  8b9: # color        
       -subtitle {upload  .          
                }
   x;
     ottom: 12prgin-b        ma
        c;olor: #f0f6f   c           600;
  ight: -we        font    em;
    ze: 1.5rsint-          fole {
      oad-tit  .upl                 
         }
  x;
      -bottom: 24p      margin         8a6ff;
  #5     color:      
     4rem;-size: font               ad-icon {
        .uplo  
     
                  }1117;
     #0dund-color:kgrobac            8a6ff;
    r: #5colo    border-    
        rea:hover {ad-a  .uplo             
 
               }
      0.3s ease;on: allnsitira      t   ;
       : 40px 0      margin    px;
      px 40ing: 60  padd            0363d;
  px dashed #3 2r:  borde         22;
     or: #161bd-colunro   backg             {
area d-   .uploa       
           
      }
         to 80px;n: 0 au   margi        x;
     00p: 6dth    max-wi           ection {
 load-s  .up         
             
     }   
    n: center;ig text-al               
x 40px;0p  padding: 8             nt {
 n-conte      .mai
                      }
        0 auto;
 rgin:     ma            600px;
ax-width:    m           b949e;
 color: #8               1.25rem;
 size: t-      fon
          eader p {        .h       
               }
;
       16pxm:margin-botto          c;
      lor: #f0f6f          co0;
       60nt-weight:     fo        em;
   -size: 3r        font        {
 ader h1    .he      
                  }
     30363d;
   d #litom: 1px sobot   border-         r;
    cente-align:      text           px;
g: 60px 40addin          p;
      r: #161b22-colokground    bac            .header {
                    
   }
          ing: 0;
     padd           uto;
    margin: 0 a            ;
   dth: 1200px  max-wi         er {
     .contain      
         }
           
           100vh;-height:   min    
         ight: 1.6;he  line-          df3;
    olor: #e6e          c    17;
   #0d11round-color:    backg            s-serif;
, sanarellUbuntu, CantOxygen, oboto, UI', R 'Segoe Font,nkMacSystem, Blitem: -apple-sys font-family             {
  dy       bo 
               }
           
   x;border-bosizing: x- bo            : 0;
    padding          ;
      0gin:        mar       * {
        
     yle>
        <sttitle>atcher</>AI Resume M    <title0">
    le=1.itial-scaindth, ice-wi=devtent="width conewport"vi name="     <meta  -8">
 TFarset="Umeta ch       <<head>
 en">
    lang="<html     PE html>
DOCTY '''
    <!
    return():ef index
d)('/'teou]

@app.r'}
ety.ssenger safensure parcraft and  aimmercialperate co'Oription': ion', 'desc 'Aviatgory':0k', 'cate'$80k-$12ry': n'], 'salacommunicatioations', 't operon', 'flighviatis': ['aill'sks',  Airline': 'Regional 'companyt',l Pilo: 'Commercia   {'title'.'},
 mprovementss iiness procesd bus advice anstrategicProvide n': ''descriptio',  'Consultantategory':k-$130k', 'calary': '$90cal'], 's, 'analytitrategy'alysis', 'sbusiness aning', '['consultskills': isors', 'gic Adv'Strateompany': ant', 'cConsultnagement 'title': 'Ma
    {ching.'},wellness coaand  training fitnessd ersonalizeovide p': 'Prption', 'descritnessgory': 'Fik', 'cate '$35k-$55ary':], 'salication'unon', 'comm, 'nutritig' trainin 'personalfitness',[': ls'kilCenter', 'se Fitness 'Elitmpany': ainer', 'corsonal Trle': 'Pe   {'titement.'},
 anagand crop mble farming nataie in susertisvide expon': 'Proriptiure', 'desc: 'Agriculty'k', 'categor50k-$70salary': '$esearch'], ' 'rfarming', 't',gemen 'crop manaculture',: ['agriskills'tions', 'SoluTech ri': 'Ag 'companyst',aliral Speci: 'Agricultu    {'title'etion.'},
ng to compl from plannion projectsructi constageanption': 'M, 'descrionstruction'tegory': 'C 'ca$75k-$105k',alary': '], 'sbudgeting'rship', 'leade', 'ction'construt',  managemenctls': ['projekilction', 'sonstrught C 'BuildRimpany':nager', 'con Project Maiotructe': 'Cons  {'titl
   menus.'},vativevelop innoons and deratitchen ope: 'Lead kin'iptiof', 'descrory': 'Che5k', 'categ'$60k-$8: y'alarership'], 'seadement', 'litchen managing', 'knn planume ' arts',ryculinalls': ['rant', 'skitauing Res'Fine Dinompany': , 'ctive Chef'xecu{'title': 'E   
 DUSTRIESER IN   # OTH,
    
 ls.'}terianding ma and brangfor marketisigns visual dete reaription': 'Csc'dener', ig 'Descategory':, '5k-$65k'salary': '$4'], 'design', 'reativen', 'cual desigsign', 'vis['graphic des': cy', 'skillrketing Ageny': 'Macompanigner', 'c DeshiGrap{'title': '
    oducts.'},al prr digitnces foxperieaces and enterfuser ign ion': 'Desiscriptigner', 'dey': 'Desategor5k', 'c': '$70k-$9lary'sa], e'n', 'creativ'desig', 'ui design', ux design ['', 'skills':gn StudioDesi'company': 'gner', UI Desie': 'UX/{'titlIGN
    
    # DESs.'},
    stemand sylications tware app soflopve': 'Deption'descri', ineeringory': 'Engeg$125k', 'cat': '$85k-'], 'salaryhon, 'pytva't', 'jaelopmenoftware devamming', 'sls': ['progrtup', 'skilch Starompany': 'Te'c Engineer', reftwaSo': 'itle  {'tts.'},
  ion projecructonstd c anastructurefrsign innd de'Plan aription': ng', 'desc'Engineeriy': , 'categork'0k-$95 '$7, 'salary':struction']ment', 'con managerojecttocad', 'pg', 'aueerin engin['civillls': ns', 'skiure Solutiotruct 'Infrasany':, 'compEngineer'': 'Civil  {'title.'},
   ng processesriufactu oversee manstems andcal syn mechaniion': 'Desig, 'descriptEngineering': 'egory'$105k', 'cat75k-y': '$ar, 'salanagement'] 'project m'design',', 'cad', neeringengi'mechanical s': ['skill', orpfacturing Canupany': 'M, 'com Engineer''Mechanicalitle':    {'t
 GINEERING# EN
    .'},
     relationsand employeetment ruiluding recs incR function various Handleiption': 'Hescr': 'HR', 'dgory 'cate',70k$50k-$ '], 'salary': 'hr'cruiting',ns', 're relatiomployeerces', 'e resouhuman'skills': [', e Company''Mid-Sizompany': 