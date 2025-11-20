"""
Prediction module for Fake News Detection System.
Handles loading trained models and making predictions on new text.
"""

import pickle
from typing import Union, List
import numpy as np

from config.config import MODEL_PATH, VECTORIZER_PATH, LABEL_RELIABLE, LABEL_UNRELIABLE
from src.preprocessing import stemming


class FakeNewsDetector:
    """
    Fake news detection class that loads trained models and makes predictions.
    
    Attributes:
        vectorizer: Loaded TF-IDF vectorizer
        model: Loaded classification model
    """
    
    def __init__(self):
        """Initialize the detector by loading trained models."""
        self.vectorizer = None
        self.model = None
        self.load_models()
    
    def load_models(self):
        """
        Load trained vectorizer and model from disk.
        
        Raises:
            FileNotFoundError: If model files are not found
        """
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                f"Please train the model first using train_model.py"
            )
        
        if not VECTORIZER_PATH.exists():
            raise FileNotFoundError(
                f"Vectorizer file not found at {VECTORIZER_PATH}. "
                f"Please train the model first using train_model.py"
            )
        
        print("Loading trained models...")
        with open(VECTORIZER_PATH, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        with open(MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)
        
        print("Models loaded successfully!")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess input text before prediction.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text
        """
        return stemming(text)
    
    def predict(self, text: str) -> int:
        """
        Predict if news article is fake or real.
        
        Args:
            text: News article text
            
        Returns:
            Prediction (0 for reliable, 1 for unreliable)
        """
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Vectorize
        input_vector = self.vectorizer.transform([processed_text])
        
        # Predict
        prediction = self.model.predict(input_vector)
        
        return int(prediction[0])
    
    def predict_proba(self, text: str) -> np.ndarray:
        """
        Get prediction probabilities for both classes.
        
        Args:
            text: News article text
            
        Returns:
            Array of probabilities [prob_reliable, prob_unreliable]
        """
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Vectorize
        input_vector = self.vectorizer.transform([processed_text])
        
        # Get probabilities (if model supports it)
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(input_vector)
            return probabilities[0]
        else:
            # For models without predict_proba, return binary predictions
            prediction = self.model.predict(input_vector)
            if prediction[0] == LABEL_RELIABLE:
                return np.array([1.0, 0.0])
            else:
                return np.array([0.0, 1.0])
    
    def is_reliable(self, text: str) -> bool:
        """
        Check if news article is reliable.
        
        Args:
            text: News article text
            
        Returns:
            True if reliable, False if unreliable
        """
        prediction = self.predict(text)
        return prediction == LABEL_RELIABLE
    
    def is_unreliable(self, text: str) -> bool:
        """
        Check if news article is unreliable/fake.
        
        Args:
            text: News article text
            
        Returns:
            True if unreliable, False if reliable
        """
        prediction = self.predict(text)
        return prediction == LABEL_UNRELIABLE
    
    def get_prediction_label(self, text: str) -> str:
        """
        Get human-readable prediction label.
        
        Args:
            text: News article text
            
        Returns:
            "Reliable" or "Unreliable"
        """
        prediction = self.predict(text)
        return "Reliable" if prediction == LABEL_RELIABLE else "Unreliable"


# Create a singleton instance for easy import
try:
    detector = FakeNewsDetector()
except FileNotFoundError as e:
    print(f"Warning: {e}")
    detector = None


def fake_news(news: str) -> List[int]:
    """
    Convenience function for fake news detection (backward compatibility).
    
    Args:
        news: News article text
        
    Returns:
        List containing prediction [0] for reliable or [1] for unreliable
    """
    if detector is None:
        raise RuntimeError("Models not loaded. Please train the model first.")
    
    prediction = detector.predict(news)
    return [prediction]


def detect_fake_news(text: str) -> dict:
    """
    Comprehensive fake news detection with detailed results.
    
    Args:
        text: News article text
        
    Returns:
        Dictionary with prediction, label, and confidence
    """
    if detector is None:
        raise RuntimeError("Models not loaded. Please train the model first.")
    
    prediction = detector.predict(text)
    label = detector.get_prediction_label(text)
    
    result = {
        'prediction': prediction,
        'label': label,
        'is_reliable': prediction == LABEL_RELIABLE,
        'is_unreliable': prediction == LABEL_UNRELIABLE
    }
    
    return result
