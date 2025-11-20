"""
Fake News Detection System - Main Application
A Streamlit web application for detecting fake news using NLP and Machine Learning.
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path for imports
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config.config import (
    PAGE_TITLE, PAGE_ICON, APP_TITLE, APP_SUBTITLE,
    TEXT_AREA_PLACEHOLDER, TEXT_AREA_HEIGHT, BUTTON_TEXT,
    MESSAGE_RELIABLE, MESSAGE_UNRELIABLE, BACKGROUND_IMAGE_PATH,
    LABEL_RELIABLE, LABEL_UNRELIABLE
)
from src.prediction import fake_news, detector
from src.utils import (
    set_background_image, set_custom_styles, validate_input,
    show_model_info, show_examples
)


# Configure Streamlit page
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """Main application function."""
    
    # Set background and custom styles
    set_background_image(BACKGROUND_IMAGE_PATH)
    set_custom_styles()
    
    # Main content
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        st.title(APP_TITLE)
        st.subheader(APP_SUBTITLE)
        
        # Check if models are loaded
        if detector is None:
            st.error(
                "⚠️ **Models not found!** Please train the model first:\n\n"
                "```bash\n"
                "python src/train_model.py\n"
                "```"
            )
            st.stop()
        
        # Text input area
        sentence = st.text_area(
            label="Enter News Article",
            value="",
            placeholder=TEXT_AREA_PLACEHOLDER,
            height=TEXT_AREA_HEIGHT,
            key="news_input"
        )
        
        # Prediction button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            predict_btn = st.button(BUTTON_TEXT, use_container_width=True)
        
        # Process prediction
        if predict_btn:
            if validate_input(sentence):
                with st.spinner('🔍 Analyzing news article...'):
                    try:
                        # Get prediction
                        prediction_class = fake_news(sentence)
                        
                        # Display result
                        st.markdown("---")
                        st.subheader("📊 Analysis Result")
                        
                        if prediction_class == [LABEL_RELIABLE]:
                            st.success(MESSAGE_RELIABLE)
                            st.balloons()
                        elif prediction_class == [LABEL_UNRELIABLE]:
                            st.warning(MESSAGE_UNRELIABLE)
                        
                        # Additional info
                        st.info(
                            "💡 **Note:** This prediction is based on machine learning analysis "
                            "and may not be 100% accurate. Always verify news from multiple reliable sources."
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
    
    # Sidebar
    show_model_info()
    
    # Show examples
    with col2:
        st.markdown("---")
        show_examples()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: white; text-shadow: 1px 1px 2px rgba(0,0,0,0.7);'>
            <p>Made with ❤️ using Python, Streamlit, and Machine Learning</p>
            <p>© 2025 Fake News Detection System</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()