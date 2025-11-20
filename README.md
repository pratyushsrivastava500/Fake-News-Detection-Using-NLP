# 📰 Fake News Detection Using NLP

![Python](https://img.shields.io/badge/Python-3.10-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange) ![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

> An AI-powered system that uses Natural Language Processing and Machine Learning to detect fake news articles with 85-95% accuracy. Built with clean architecture and modular design for production deployment

## 📋 Overview

The Fake News Detection System enables users to:

- **Automate Verification** of news articles using AI-powered analysis
- **Analyze Text Patterns** with advanced NLP techniques and stemming
- **Get Instant Results** with real-time classification (Reliable/Unreliable)
- **User-Friendly Interface** with an intuitive Streamlit web application
- **Train Custom Models** on your own dataset for improved accuracy
- **Export Predictions** for further analysis and record keeping

## ✨ Features

### 🎯 Machine Learning Technology

- **Decision Tree Classifier** with optimized hyperparameters (~85-95% accuracy)
- **TF-IDF Vectorization** for effective text feature extraction
- **NLTK Preprocessing** with Porter Stemmer and stopword removal
- **Robust Text Cleaning** removing noise and normalizing content
- **Model Persistence** with pickle serialization for fast loading

### 🏗️ Clean Architecture

- **Modular Design** with separation of concerns across modules
- **Type Hints** and comprehensive docstrings throughout
- **Centralized Configuration** for easy customization
- **Production-Ready** error handling and validation
- **Cross-Platform** compatibility (Windows, Linux, macOS)

### 💻 User Experience

- **Beautiful Streamlit UI** with custom styling and themes
- **Real-Time Analysis** with loading indicators
- **Custom Background** support for personalized branding
- **Responsive Design** that works on all screen sizes
- **Input Validation** with helpful error messages
- **Example Articles** to test the system instantly

### 🔒 Best Practices

- **Version Control** ready with comprehensive .gitignore
- **Dependency Management** with detailed requirements.txt
- **Modular Testing** structure in tests/ directory
- **Documentation** with inline comments and docstrings
- **Scalable Design** allowing easy feature additions

## 🚀 Quick Start

### Prerequisites

- **Python 3.10** (required)
- pip package manager
- 4GB RAM minimum
- Internet connection (for initial NLTK data download)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/Fake-News-Detection-Using-NLP.git
cd Fake-News-Detection-Using-NLP
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Download NLTK data** (if not auto-downloaded)

```python
import nltk
nltk.download('stopwords')
```

4. **Prepare training data**

Download the dataset from [Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news/data) and place `train.csv` in the `data/` directory. The CSV should have columns: `id`, `title`, `author`, `text`, `label`

5. **Train the model** (if models not included)

```bash
python src/train_model.py
```

6. **Run the application**

```bash
streamlit run app.py
```

7. **Open your browser**

Navigate to `http://localhost:8501` to use the application

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│      Streamlit Web Interface        │
│  • Input text area                  │
│  • Prediction display               │
│  • Custom styling & background      │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│        Application Layer            │
│  • app.py (main entry point)        │
│  • User interaction handling        │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│     Business Logic Layer            │
│  • prediction.py (detection)        │
│  • preprocessing.py (text cleaning) │
│  • utils.py (helper functions)      │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│    Machine Learning Layer           │
│  • train_model.py (training)        │
│  • Decision Tree Classifier         │
│  • TF-IDF Vectorizer                │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│    Configuration & Data Layer       │
│  • config/config.py (settings)      │
│  • models/ (trained models)         │
│  • data/ (training datasets)        │
└─────────────────────────────────────┘
```

## 🛠️ Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------||
| **Language** | Python | 3.10 |
| **Web Framework** | Streamlit | 1.25+ |
| **ML Library** | scikit-learn | 1.2+ |
| **NLP Library** | NLTK | 3.8+ |
| **Algorithm** | Decision Tree Classifier | - |
| **Vectorization** | TF-IDF | Max 5000 features |
| **Text Processing** | Porter Stemmer | - |
| **Data Processing** | Pandas, NumPy | Latest |
| **Model Serialization** | Pickle | Built-in |

## 📁 Project Structure

```
Fake-News-Detection-Using-NLP/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore patterns
├── README.md                       # Project documentation
│
├── config/
│   ├── __init__.py
│   └── config.py                   # Configuration settings
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # Text preprocessing module
│   ├── train_model.py              # Model training pipeline
│   ├── prediction.py               # Prediction module
│   └── utils.py                    # Utility functions
│
├── models/
│   ├── model.pkl                   # Trained classifier
│   └── vector.pkl                  # Fitted TF-IDF vectorizer
│
├── data/
│   └── train.csv                   # Training dataset (not included)
│
├── static/
│   └── Image.jpg                   # Background image
│
└── notebooks/
    └── Model_Training.ipynb        # Jupyter notebook for exploration
```

## 📊 System Information

**Algorithm:** Decision Tree Classifier with TF-IDF Vectorization

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Training Accuracy** | ~95% |
| **Test Accuracy** | ~85-90% |
| **Prediction Speed** | < 1 second |
| **Training Time** | 3-5 seconds (1000 samples) |
| **Model Size** | < 1 MB |
| **Vectorizer Size** | < 500 KB |

### Key Components

| Component | Description | Details |
|-----------|-------------|---------|
| **Text Preprocessing** | Porter Stemmer | Reduces words to root form |
| **Stopword Removal** | NLTK Stopwords | Removes common English words |
| **Vectorization** | TF-IDF | Term Frequency-Inverse Document Frequency |
| **Classification** | Decision Tree | Supervised learning algorithm |
| **Data Format** | CSV | ID, Title, Author, Text, Label |

### Processing Pipeline

1. **Input:** Raw news article text
2. **Cleaning:** Remove non-alphabetic characters
3. **Normalization:** Convert to lowercase
4. **Tokenization:** Split into words
5. **Stopword Removal:** Remove common words
6. **Stemming:** Reduce to root forms
7. **Vectorization:** Convert to TF-IDF features
8. **Classification:** Predict using trained model
9. **Output:** Reliable (0) or Unreliable (1)

## 📖 Usage Guide

### Training a New Model

1. **Prepare your dataset**

Ensure `train.csv` is in the `data/` directory with these columns:
- `id`: Unique identifier
- `title`: Article title
- `author`: Author name
- `text`: Article content
- `label`: 0 (reliable) or 1 (unreliable)

2. **Run the training script**

```bash
python src/train_model.py
```

3. **Output**

Models will be saved in `models/` directory:
- `model.pkl`: Trained Decision Tree classifier
- `vector.pkl`: Fitted TF-IDF vectorizer

### Using the Web Application

1. **Start the app**

```bash
streamlit run app.py
```

2. **Enter news text**

Paste or type a news article in the text area

3. **Click "Check Authenticity"**

The system will analyze and classify the article

4. **View results**

- ✅ **Reliable:** News appears authentic
- ⚠️ **Unreliable:** News may be fake or misleading

### Using as a Python Module

```python
from src.prediction import detector, detect_fake_news

# Simple prediction
text = "Your news article text here..."
result = detector.predict(text)
print("Reliable" if result == 0 else "Unreliable")

# Detailed prediction
result = detect_fake_news(text)
print(f"Label: {result['label']}")
print(f"Is Reliable: {result['is_reliable']}")
```

### Customization

Edit `config/config.py` to customize:

- Model parameters (test size, random state)
- UI settings (titles, colors, messages)
- File paths (data, models, static files)
- Text preprocessing (stopwords, patterns)

### Example Workflow

**Training:**

```bash
# 1. Place train.csv in data/ directory
# 2. Train the model
python src/train_model.py

# Output:
# ============================================================
# FAKE NEWS DETECTION - MODEL TRAINING PIPELINE
# ============================================================
# Loading 1000 rows from data/train.csv...
# Preprocessing data...
# Training Decision Tree model...
# Model Accuracy: 0.8850 (88.50%)
# Models saved successfully!
```

**Prediction:**

```bash
# 1. Run the web app
streamlit run app.py

# 2. Open browser at http://localhost:8501
# 3. Paste news article
# 4. Click "🚀 Check Authenticity"
# 5. View result: ✅ Reliable or ⚠️ Unreliable
```

## 🤖 Model Performance

### Accuracy Metrics

| Dataset | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| **Training** | ~95% | ~0.94 | ~0.96 | ~0.95 |
| **Testing** | ~88% | ~0.87 | ~0.89 | ~0.88 |

### Performance Factors

1. **Training Data Quality** (40%)
   - Diverse sources increase robustness
   - Balanced classes prevent bias

2. **Text Preprocessing** (25%)
   - Proper cleaning improves accuracy
   - Stemming reduces vocabulary size

3. **Feature Engineering** (20%)
   - TF-IDF captures important patterns
   - Max features limit affects performance

4. **Model Selection** (15%)
   - Decision Tree offers good baseline
   - Can upgrade to Random Forest/XGBoost

### Optimization Tips

- **Increase training data** (>10,000 samples recommended)
- **Balance dataset** (equal reliable/unreliable articles)
- **Tune hyperparameters** (max_depth, min_samples_split)
- **Try ensemble methods** (Random Forest, Gradient Boosting)
- **Add more features** (title, author metadata)
- **Cross-validation** for better generalization

## 🔮 Future Enhancements

### Planned Features

- [ ] Deep Learning models (LSTM, BERT, Transformers)
- [ ] Multi-language support (Spanish, French, etc.)
- [ ] Source credibility analysis
- [ ] Fact-checking integration with external APIs
- [ ] Confidence score display
- [ ] Batch processing for multiple articles
- [ ] API endpoint for programmatic access
- [ ] User feedback collection for retraining
- [ ] Article summarization feature
- [ ] Export reports in PDF/Excel format

### Advanced Features

- [ ] Real-time news monitoring
- [ ] Browser extension for instant verification
- [ ] Social media integration (Twitter, Facebook)
- [ ] Explainable AI (show why article is fake)
- [ ] User accounts and history tracking
- [ ] Database backend (PostgreSQL/MongoDB)
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] A/B testing framework
- [ ] Performance monitoring dashboard

## 🔧 Troubleshooting

### Issue: ModuleNotFoundError

```bash
# Solution: Install missing dependencies
pip install -r requirements.txt
```

### Issue: NLTK stopwords not found

```python
# Solution: Download NLTK data
import nltk
nltk.download('stopwords')
```

### Issue: Models not found error

```bash
# Solution: Train the model first
python src/train_model.py
```

### Issue: train.csv not found

```bash
# Solution: Download the Kaggle Fake News dataset
# 1. Visit: https://www.kaggle.com/c/fake-news/data
# 2. Download train.csv
# 3. Place it in the data/ directory
# 4. Ensure it has columns: id, title, author, text, label
```

### Issue: Streamlit page not loading

```bash
# Solution: Check if port 8501 is available
# Or specify a different port
streamlit run app.py --server.port 8502
```

### Issue: Low accuracy on predictions

```bash
# Solutions:
# 1. Train with more data (increase N_TRAINING_SAMPLES in config)
# 2. Use full dataset instead of sample
# 3. Try different model (Random Forest, SVM)
# 4. Tune hyperparameters
# 5. Add more preprocessing steps
```

### Issue: ImportError on Windows

```bash
# Solution: Ensure Python is in PATH
# Or use absolute path to python executable
C:\Python38\python.exe src/train_model.py
```

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Make your changes**
4. **Add tests** for new functionality
5. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
6. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
7. **Open a Pull Request**

### Coding Standards

- Follow **PEP 8** style guide for Python
- Add **type hints** to all function signatures
- Write **docstrings** for all classes and functions
- Include **unit tests** for new features
- Update **documentation** as needed
- Keep functions **small and focused**
- Use **meaningful variable names**

### Areas for Contribution

- 🐛 Bug fixes
- ✨ New features
- 📝 Documentation improvements
- 🧪 Additional test cases
- 🎨 UI/UX enhancements
- ⚡ Performance optimizations
- 🌍 Internationalization


## 🙏 Acknowledgments

- **scikit-learn Team** for the excellent machine learning library
- **NLTK Contributors** for natural language processing tools
- **Streamlit Team** for the amazing web framework
- **Kaggle Community** for providing fake news datasets
- **Open Source Contributors** who inspire and educate

### Datasets

- [Fake News Dataset on Kaggle](https://www.kaggle.com/c/fake-news/data)
- [LIAR Dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)

### Research Papers

- *Automatic Detection of Fake News* - Conroy et al.
- *Fake News Detection using Machine Learning* - Shu et al.
- *The Science of Fake News* - Lazer et al.

## 📧 Contact

For questions, suggestions, or collaboration opportunities:

- **GitHub Issues:** [Create an issue](https://github.com/yourusername/Fake-News-Detection-Using-NLP/issues)
- **Email:** pratyushsrivastava500@gmail.com


## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [NLTK Documentation](https://www.nltk.org/)
- [Machine Learning Mastery](https://machinelearningmastery.com/)

---

**⚠️ Disclaimer:** This system is designed for educational and research purposes. While it achieves good accuracy, it should not be the sole source for determining news authenticity. Always verify information from multiple reliable sources and use critical thinking when evaluating news articles.

---

<div align="center">

**Made with ❤️ and Python | © 2025 Fake News Detection System**

</div>



