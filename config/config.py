"""
Configuration file for Fake News Detection System.
Contains all paths, parameters, and settings used throughout the application.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Directory paths
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Model paths
MODEL_PATH = MODELS_DIR / "model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vector.pkl"

# Data paths
TRAIN_DATA_PATH = DATA_DIR / "train.csv"

# Static assets
BACKGROUND_IMAGE_PATH = STATIC_DIR / "Image.jpg"

# Model parameters
TEST_SIZE = 0.20
RANDOM_STATE = 42
MAX_FEATURES = 5000
N_TRAINING_SAMPLES = 1000  # Number of rows to load from training data

# Streamlit UI configuration
PAGE_TITLE = "AI-Powered Fake News Analyzer"
PAGE_ICON = "📰"
APP_TITLE = "AI-Powered Fake News Analyzer"
APP_SUBTITLE = "Is This News Real? Enter Below"
TEXT_AREA_PLACEHOLDER = "Paste Your News Article Here"
TEXT_AREA_HEIGHT = 200
BUTTON_TEXT = "🚀 Check Authenticity"

# Classification labels
LABEL_RELIABLE = 0
LABEL_UNRELIABLE = 1

# Result messages
MESSAGE_RELIABLE = "✅ Reliable - This news appears to be authentic"
MESSAGE_UNRELIABLE = "⚠️ Unreliable - This news may be fake or misleading"

# Text preprocessing settings
STOPWORDS_LANGUAGE = "english"
PATTERN_NON_ALPHA = r'[^a-zA-Z]'
REPLACEMENT_CHAR = ' '

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, STATIC_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
