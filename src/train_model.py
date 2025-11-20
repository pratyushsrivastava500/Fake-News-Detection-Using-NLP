"""
Model training module for Fake News Detection System.
Handles data loading, preprocessing, model training, and persistence.
"""

import pandas as pd
import pickle
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk

from config.config import (
    TRAIN_DATA_PATH, MODEL_PATH, VECTORIZER_PATH,
    TEST_SIZE, RANDOM_STATE, MAX_FEATURES, N_TRAINING_SAMPLES
)
from src.preprocessing import stemming


def download_nltk_resources():
    """Download required NLTK resources."""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)


def load_data(nrows: int = N_TRAINING_SAMPLES) -> pd.DataFrame:
    """
    Load training data from CSV file.
    
    Args:
        nrows: Number of rows to load (default from config)
        
    Returns:
        DataFrame with training data
        
    Raises:
        FileNotFoundError: If training data file not found
    """
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found at {TRAIN_DATA_PATH}. "
            f"Please place train.csv in the data/ directory."
        )
    
    print(f"Loading {nrows} rows from {TRAIN_DATA_PATH}...")
    df = pd.read_csv(TRAIN_DATA_PATH, nrows=nrows)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset: fill nulls, remove unnecessary columns, apply stemming.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    print("Preprocessing data...")
    
    # Fill null values
    df = df.fillna('')
    
    # Remove unnecessary columns
    columns_to_drop = ['id', 'title', 'author']
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    if existing_columns:
        df = df.drop(existing_columns, axis=1)
    
    # Apply stemming to text
    print("Applying text preprocessing (stemming, stopword removal)...")
    df['text'] = df['text'].apply(stemming)
    
    print("Preprocessing completed.")
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Separate input features (X) and target labels (y).
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        Tuple of (X, y) where X is text data and y is labels
    """
    X = df['text']
    y = df['label']
    return X, y


def split_data(X: pd.Series, y: pd.Series) -> Tuple:
    """
    Split data into training and testing sets.
    
    Args:
        X: Feature data
        y: Target labels
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    print(f"Splitting data (test_size={TEST_SIZE})...")
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def vectorize_text(X_train: pd.Series, X_test: pd.Series) -> Tuple:
    """
    Convert text to TF-IDF vectors.
    
    Args:
        X_train: Training text data
        X_test: Testing text data
        
    Returns:
        Tuple of (X_train_vec, X_test_vec, vectorizer)
    """
    print("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"Vectorization completed. Feature shape: {X_train_vec.shape}")
    return X_train_vec, X_test_vec, vectorizer


def train_model(X_train, y_train):
    """
    Train Decision Tree classifier.
    
    Args:
        X_train: Training feature vectors
        y_train: Training labels
        
    Returns:
        Trained model
    """
    print("Training Decision Tree model...")
    model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    print("Model training completed.")
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_test: Testing features
        y_test: Testing labels
        
    Returns:
        Model accuracy score
    """
    print("\nEvaluating model...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=['Reliable', 'Unreliable']))
    
    return accuracy


def save_models(vectorizer, model):
    """
    Save trained vectorizer and model to disk.
    
    Args:
        vectorizer: Trained TF-IDF vectorizer
        model: Trained classification model
    """
    print(f"\nSaving models to {MODELS_DIR}...")
    
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer saved to {VECTORIZER_PATH}")
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_PATH}")
    
    print("Models saved successfully!")


def train_pipeline():
    """
    Complete training pipeline: load data, preprocess, train, evaluate, save.
    """
    print("=" * 60)
    print("FAKE NEWS DETECTION - MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Download NLTK resources
    download_nltk_resources()
    
    # Load and preprocess data
    df = load_data()
    df = preprocess_data(df)
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Vectorize text
    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)
    
    # Train model
    model = train_model(X_train_vec, y_train)
    
    # Evaluate model
    accuracy = evaluate_model(model, X_test_vec, y_test)
    
    # Save models
    save_models(vectorizer, model)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return model, vectorizer, accuracy


if __name__ == "__main__":
    train_pipeline()
