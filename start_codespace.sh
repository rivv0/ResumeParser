#!/bin/bash

echo "🚀 Starting ML Resume Matcher in GitHub Codespace"
echo "=================================================="

# Check if requirements are installed
if [ ! -d "venv" ]; then
    echo "📦 Setting up virtual environment..."
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Create uploads directory
mkdir -p uploads

# Pre-download the ML model to avoid timeout on first request
echo "🤖 Pre-loading ML model..."
python -c "
from sentence_transformers import SentenceTransformer
print('Loading model...')
model = SentenceTransformer('all-MiniLM-L6-v2')
print('Model loaded successfully!')
"

echo "✅ Setup complete!"
echo ""
echo "🌐 Starting the application..."
echo "📍 Your app will be available at the forwarded port 5009"
echo "🔗 GitHub Codespaces will automatically open the browser"
echo ""

# Start the Flask app
python app.py