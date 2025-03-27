#!/bin/bash
# scripts/setup_env.sh

# Initialize project environment
set -e

echo "🚀 Setting up FinGenius environment..."

# Verify Python version
REQ_PYTHON="3.9.0"
CUR_PYTHON=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
if [ "$(printf '%s\n' "$REQ_PYTHON" "$CUR_PYTHON" | sort -V | head -n1)" != "$REQ_PYTHON" ]; then
    echo "❌ Python $REQ_PYTHON+ required. Found $CUR_PYTHON"
    exit 1
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "💻 Creating virtual environment..."
    python3 -m venv venv
fi

# Install dependencies
echo "📦 Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Download NLP models
echo "🔧 Downloading NLP models..."
python -m spacy download en_core_web_sm

# Setup environment variables
if [ ! -f ".env" ]; then
    echo "🔒 Creating .env template..."
    cp .env.example .env
    echo "⚠️  Update .env file with your API keys!"
fi

echo "✅ Setup complete! Activate with: source venv/bin/activate"
