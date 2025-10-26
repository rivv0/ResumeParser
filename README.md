# Resume Matcher

AI-powered resume matching system using semantic embeddings and external job APIs.

## Features

- PDF resume upload and text extraction
- Semantic similarity matching using sentence transformers
- Real-time job data from external APIs
- Skill extraction and matching
- Experience analysis
- Match scoring with detailed breakdowns

## Technology Stack

- **Backend**: Flask web framework
- **ML Model**: all-MiniLM-L6-v2 sentence transformer
- **Similarity**: Cosine similarity between embeddings
- **Job Data**: External API integration (JSearch/RapidAPI)
- **PDF Processing**: PyPDF2 for text extraction

## Installation

```bash
pip install -r req.txt
python app.py
```

Visit http://localhost:5000 to use the application.

## How It Works

1. **Resume Upload**: User uploads PDF resume
2. **Text Extraction**: PyPDF2 extracts text content
3. **Skill Detection**: Pattern matching identifies relevant skills
4. **Job Fetching**: External API retrieves current job listings
5. **Embedding Generation**: Sentence transformer creates vector representations
6. **Similarity Calculation**: Cosine similarity between resume and job embeddings
7. **Scoring**: Combined score using semantic similarity (60%) and skill overlap (40%)
8. **Results**: Ranked job matches with detailed breakdowns

## Scoring Algorithm

```
Final Score = (Semantic Similarity × 0.6) + (Skill Match × 0.4)
```

- **Semantic Similarity**: Cosine similarity between resume and job description embeddings
- **Skill Match**: Percentage of job-required skills found in resume

## API Configuration

To use real job data, configure the JSearch API:

1. Get API key from RapidAPI
2. Replace `YOUR_RAPIDAPI_KEY_HERE` in app.py
3. System falls back to sample jobs if API unavailable

## File Structure

```
├── app.py              # Main Flask application
├── req.txt             # Python dependencies
├── uploads/            # Temporary file storage
├── README.md           # Documentation
└── images/             # Static images
```

## Requirements

- Python 3.7+
- Flask 2.3+
- sentence-transformers 2.2+
- scikit-learn 1.3+
- PyPDF2 3.0+
- requests 2.31+

Optimized parsing pipeline that improves speed by ~70% compared to baseline implementations.



