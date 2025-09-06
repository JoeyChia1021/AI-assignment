#!/bin/bash

echo "🔍 Starting Shape Recognition System..."
echo "📦 Using virtual environment: .venv"
echo "🌐 The app will open at: http://localhost:8501"
echo ""

# Activate virtual environment and run streamlit
source .venv/bin/activate
streamlit run shape_recognizer.py
