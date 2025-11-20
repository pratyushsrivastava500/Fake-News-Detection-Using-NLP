"""
Utility functions for Fake News Detection System.
"""

import base64
from pathlib import Path
from typing import Optional
import streamlit as st


def set_background_image(image_path: Path) -> None:
    """
    Set a background image for the Streamlit app.
    
    Args:
        image_path: Path to the background image file
    """
    if not image_path.exists():
        print(f"Warning: Background image not found at {image_path}")
        return
    
    try:
        with open(image_path, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode()
        
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded_string}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        print(f"Error setting background image: {e}")


def set_custom_styles() -> None:
    """Apply custom CSS styles to the Streamlit app."""
    st.markdown(
        """
        <style>
        h1, h2, h3 {
            color: white !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.7);
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            padding: 12px 24px;
            border-radius: 8px;
            border: none;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #45a049;
            box-shadow: 0 6px 8px rgba(0,0,0,0.15);
            transform: translateY(-2px);
        }
        .stTextArea>div>div>textarea {
            font-size: 16px;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def display_prediction_result(prediction: int, label: str) -> None:
    """
    Display prediction result with appropriate styling.
    
    Args:
        prediction: Prediction value (0 or 1)
        label: Human-readable label
    """
    if prediction == 0:
        st.success(f'✅ {label}')
        st.balloons()
    else:
        st.warning(f'⚠️ {label}')


def validate_input(text: str) -> bool:
    """
    Validate input text before prediction.
    
    Args:
        text: Input text to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not text or text.strip() == "" or text.strip() == "Paste Your News Article Here":
        st.error("❌ Please enter some news text to analyze!")
        return False
    
    if len(text.strip()) < 20:
        st.warning("⚠️ Please enter a longer text for better accuracy (at least 20 characters).")
        return False
    
    return True


def format_confidence(probability: float) -> str:
    """
    Format confidence score as percentage.
    
    Args:
        probability: Probability value (0-1)
        
    Returns:
        Formatted percentage string
    """
    return f"{probability * 100:.2f}%"


def show_model_info() -> None:
    """Display information about the model in the sidebar."""
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(
            """
            This AI-powered system uses Natural Language Processing (NLP) 
            and Machine Learning to detect fake news articles.
            
            **Technology Stack:**
            - Decision Tree Classifier
            - TF-IDF Vectorization
            - NLTK for text preprocessing
            - Porter Stemmer algorithm
            
            **How it works:**
            1. Enter a news article
            2. Text is cleaned and preprocessed
            3. Model analyzes linguistic patterns
            4. Prediction is made based on trained data
            """
        )
        
        st.header("📊 Model Stats")
        st.metric("Algorithm", "Decision Tree")
        st.metric("Accuracy", "~85-95%")
        st.metric("Features", "TF-IDF Vectors")


def show_examples() -> None:
    """Display example news articles in an expander."""
    with st.expander("📰 See Example Articles"):
        st.subheader("Reliable News Example")
        st.text_area(
            "Example 1",
            "The stock market closed higher today as investors reacted positively to "
            "the latest economic data. The Dow Jones Industrial Average rose 300 points, "
            "while the S&P 500 gained 1.2%. Tech stocks led the rally.",
            height=100,
            key="example1",
            disabled=True
        )
        
        st.subheader("Potentially Fake News Example")
        st.text_area(
            "Example 2",
            "BREAKING: Scientists discover aliens living among us! Government has been "
            "hiding the truth for decades. Share this before it gets deleted!",
            height=100,
            key="example2",
            disabled=True
        )
