"""
Text preprocessing module for Fake News Detection System.
Handles text cleaning, stemming, and stopword removal.
"""

import re
from typing import List
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

from config.config import STOPWORDS_LANGUAGE, PATTERN_NON_ALPHA, REPLACEMENT_CHAR


class TextPreprocessor:
    """
    Text preprocessing class that handles text cleaning and stemming.
    
    Attributes:
        stemmer: PorterStemmer instance for word stemming
        stop_words: Set of English stopwords
    """
    
    def __init__(self):
        """Initialize the text preprocessor with stemmer and stopwords."""
        self.stemmer = PorterStemmer()
        
        # Download stopwords if not already present
        try:
            self.stop_words = set(stopwords.words(STOPWORDS_LANGUAGE))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words(STOPWORDS_LANGUAGE))
    
    def clean_text(self, content: str) -> str:
        """
        Remove non-alphabetic characters from text.
        
        Args:
            content: Input text string
            
        Returns:
            Cleaned text with only alphabetic characters and spaces
        """
        return re.sub(PATTERN_NON_ALPHA, REPLACEMENT_CHAR, content)
    
    def remove_stopwords(self, words: List[str]) -> List[str]:
        """
        Remove stopwords from a list of words.
        
        Args:
            words: List of words
            
        Returns:
            List of words without stopwords
        """
        return [word for word in words if word not in self.stop_words]
    
    def stem_words(self, words: List[str]) -> List[str]:
        """
        Apply stemming to a list of words.
        
        Args:
            words: List of words to stem
            
        Returns:
            List of stemmed words
        """
        return [self.stemmer.stem(word) for word in words]
    
    def preprocess(self, content: str) -> str:
        """
        Complete preprocessing pipeline: clean, lowercase, tokenize, remove stopwords, stem.
        
        Args:
            content: Raw text content
            
        Returns:
            Preprocessed text string
        """
        # Clean text (remove non-alphabetic characters)
        cleaned = self.clean_text(content)
        
        # Convert to lowercase
        cleaned = cleaned.lower()
        
        # Tokenize (split into words)
        words = cleaned.split()
        
        # Remove stopwords and stem
        words = self.remove_stopwords(words)
        words = self.stem_words(words)
        
        # Join back into string
        return ' '.join(words)


# Create a singleton instance for easy import
preprocessor = TextPreprocessor()


def stemming(content: str) -> str:
    """
    Convenience function for text preprocessing.
    
    Args:
        content: Raw text content
        
    Returns:
        Preprocessed text string
    """
    return preprocessor.preprocess(content)
