#!/bin/bash

echo "🚀 ML Resume Matcher Deployment Script"
echo "======================================"

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit for deployment"
fi

echo ""
echo "Choose deployment option:"
echo "1. Heroku"
echo "2. Docker (local)"
echo "3. Railway (manual setup required)"
echo "4. Render (manual setup required)"
echo ""

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "🔧 Setting up Heroku deployment..."
        
        # Check if Heroku CLI is installed
        if ! command -v heroku &> /dev/null; then
            echo "❌ Heroku CLI not found. Please install it first:"
            echo "https://devcenter.heroku.com/articles/heroku-cli"
            exit 1
        fi
        
        # Login check
        if ! heroku auth:whoami &> /dev/null; then
            echo "Please login to Heroku first:"
            heroku login
        fi
        
        read -p "Enter your Heroku app name: " app_name
        
        # Create Heroku app
        heroku create $app_name
        
        # Set environment variables
        heroku config:set FLASK_ENV=production --app $app_name
        
        # Add Heroku remote if not exists
        if ! git remote get-url heroku &> /dev/null; then
            heroku git:remote -a $app_name
        fi
        
        # Deploy
        git add .
        git commit -m "Deploy to Heroku" || echo "No changes to commit"
        git push heroku main
        
        echo "✅ Deployment complete!"
        echo "🌐 Opening your app..."
        heroku open --app $app_name
        ;;
        
    2)
        echo "🐳 Setting up Docker deployment..."
        
        # Check if Docker is installed
        if ! command -v docker &> /dev/null; then
            echo "❌ Docker not found. Please install Docker first:"
            echo "https://docs.docker.com/get-docker/"
            exit 1
        fi
        
        # Build and run
        echo "Building Docker image..."
        docker build -t resume-matcher .
        
        echo "Starting container..."
        docker run -d -p 5009:5009 --name resume-matcher-app resume-matcher
        
        echo "✅ Deployment complete!"
        echo "🌐 Your app is running at: http://localhost:5009"
        ;;
        
    3)
        echo "🚂 Railway deployment setup:"
        echo "1. Go to https://railway.app"
        echo "2. Connect your GitHub repository"
        echo "3. Railway will auto-detect the Python app"
        echo "4. Set environment variable: FLASK_ENV=production"
        echo "5. Deploy automatically"
        ;;
        
    4)
        echo "🎨 Render deployment setup:"
        echo "1. Go to https://render.com"
        echo "2. Connect your GitHub repository"
        echo "3. Create a new Web Service"
        echo "4. Build Command: pip install -r requirements.txt"
        echo "5. Start Command: gunicorn app:app --bind 0.0.0.0:\$PORT --timeout 120"
        echo "6. Set environment variable: FLASK_ENV=production"
        ;;
        
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "📚 For detailed deployment instructions, check DEPLOYMENT.md"