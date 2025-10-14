#!/usr/bin/env python3
"""
Setup script for the Advanced ML Resume Matcher
This script helps users set up the system with all required dependencies
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_requirements():
    """Install Python requirements"""
    requirements_files = ['ml_requirements.txt', 'req.txt']
    
    for req_file in requirements_files:
        if os.path.exists(req_file):
            if not run_command(f"pip install -r {req_file}", f"Installing requirements from {req_file}"):
                return False
    
    return True

def download_spacy_model():
    """Download spaCy English model"""
    return run_command("python -m spacy download en_core_web_sm", "Downloading spaCy English model")

def create_directories():
    """Create necessary directories"""
    directories = ['uploads', 'models', 'data']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def test_ml_system():
    """Test if the ML system can be imported and initialized"""
    print("🧪 Testing ML system initialization...")
    
    try:
        # Test basic imports
        import torch
        import transformers
        from sentence_transformers import SentenceTransformer
        import sklearn
        import spacy
        
        print("✅ All required libraries imported successfully")
        
        # Test spaCy model
        try:
            nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy English model loaded successfully")
        except OSError:
            print("⚠️ spaCy English model not found, but can be downloaded later")
        
        # Test if CUDA is available (optional)
        if torch.cuda.is_available():
            print(f"✅ CUDA is available with {torch.cuda.device_count()} GPU(s)")
        else:
            print("ℹ️ CUDA not available, will use CPU (slower but functional)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Advanced ML Resume Matcher Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Install requirements
    print("\n📦 Installing Python packages...")
    if not install_requirements():
        print("❌ Failed to install requirements. Please check your internet connection and try again.")
        sys.exit(1)
    
    # Download spaCy model
    print("\n🔤 Setting up NLP models...")
    if not download_spacy_model():
        print("⚠️ spaCy model download failed, but you can try again later with:")
        print("   python -m spacy download en_core_web_sm")
    
    # Test the system
    print("\n🧪 Testing system...")
    if test_ml_system():
        print("\n✅ Setup completed successfully!")
        print("\n🎯 Next steps:")
        print("1. Run the ML system: python ml_resume_matcher.py")
        print("2. Or start the web app: python ml_web_app.py")
        print("3. Visit http://localhost:5007 in your browser")
        print("\n📚 Note: First run may take 2-3 minutes to download ML models")
    else:
        print("\n❌ Setup completed with errors. Please check the error messages above.")
        print("You may need to install additional dependencies or check your Python environment.")

if __name__ == "__main__":
    main()