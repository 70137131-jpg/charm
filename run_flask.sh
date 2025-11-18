#!/bin/bash
# Simple script to run the Flask application

echo "Starting RAG Pipeline Flask Server..."
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found!"
    echo "Creating .env from .env.example..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "Please edit .env file with your API keys before running queries."
    else
        echo "Please create a .env file with your configuration."
    fi
    echo ""
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: No virtual environment detected."
    echo "Consider activating a virtual environment first:"
    echo "  python -m venv venv"
    echo "  source venv/bin/activate  # On Linux/Mac"
    echo "  venv\\Scripts\\activate     # On Windows"
    echo ""
fi

# Run the Flask application
python app.py "$@"
