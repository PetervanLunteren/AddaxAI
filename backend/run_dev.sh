#!/bin/bash
# Development server launcher

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Run: python3.13 -m venv venv"
    exit 1
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Error: .env file not found. Copy .env.example to .env and configure it."
    exit 1
fi

# Activate venv and run server
source venv/bin/activate
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
