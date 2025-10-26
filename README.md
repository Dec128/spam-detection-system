# Spam Detection System

A machine learning-based spam detection system that accurately classifies emails as spam or normal (ham) using Python, Scikit-learn, and NLP libraries. The system includes user feedback capabilities and adaptive learning to improve accuracy over time.

## 🎯 Project Overview

This project was developed as part of a computer science course assignment, focusing on creating an efficient spam detection tool that addresses the growing problem of email spam and cyber harassment. The system prioritizes user feedback and adaptability to create a safer digital ecosystem.

### Team Members
- Alice Johnson - 1234
- Bob Smith - 5678
- Clara Lee - 9012
- David Kim - 3456

### Key Features
- **Advanced Text Preprocessing**: Cleans, tokenizes, and processes email text using NLTK and spaCy
- **Multiple ML Algorithms**: Supports Naive Bayes, SVM, Random Forest, and Logistic Regression
- **Feature Extraction**: Uses TF-IDF and Bag-of-Words for numerical feature representation
- **User Feedback System**: Collects user corrections to improve model accuracy
- **Adaptive Learning**: Automatically retrains models based on user feedback
- **User-Friendly GUI**: Intuitive interface for email classification and feedback
- **Comprehensive Testing**: Full test suite for all modules

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd spam-detection-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (if not automatically downloaded)
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')
   ```

4. **Download spaCy model** (optional, for better lemmatization)
   ```bash
   python -m spacy download en_core_web_sm
   ```

### Running the Application

1. **Train the model** (first time only)
   ```bash
   python train_model.py
   ```

2. **Launch the GUI application**
   ```bash
   python spam_detection_system/main.py
   ```

## 📁 Project Structure

```
spam-detection-system/
├── spam_detection_system/          # Main application package
│   ├── __init__.py
│   └── main.py                     # GUI application
├── modules/                        # Core system modules
│   ├── __init__.py
│   ├── preprocessing.py            # Text preprocessing
│   ├── feature_extraction.py       # Feature extraction
│   ├── classification.py           # ML classification
│   └── feedback_retraining.py      # Feedback and retraining
├── data/                          # Data and model storage
│   ├── sample_emails.csv          # Sample training data
│   ├── user_feedback.json         # User feedback storage
│   └── *.pkl                      # Saved models and components
├── tests/                         # Test suite
│   └── test_modules.py            # Comprehensive tests
├── requirements.txt               # Python dependencies
├── setup.py                      # Package setup
├── train_model.py                # Model training script
└── README.md                     # This file
```

## 🔧 System Architecture

### Core Modules

1. **Preprocessing Module** (`modules/preprocessing.py`)
   - Text cleaning and normalization
   - Tokenization and stop word removal
   - Stemming/lemmatization using NLTK and spaCy
   - Custom spam-specific stop words

2. **Feature Extraction Module** (`modules/feature_extraction.py`)
   - TF-IDF vectorization
   - Bag-of-Words representation
   - Feature selection using chi-squared test
   - N-gram support (1-2 grams)

3. **Classification Module** (`modules/classification.py`)
   - Multiple ML algorithms (Naive Bayes, SVM, Random Forest, Logistic Regression)
   - Model evaluation and cross-validation
   - Hyperparameter optimization
   - Model persistence (save/load)

4. **Feedback and Retraining Module** (`modules/feedback_retraining.py`)
   - User feedback collection
   - Model retraining based on feedback
   - Performance tracking and statistics
   - Adaptive learning system

### Data Flow

```
Raw Email Text
    ↓
Preprocessing Module (cleaning, tokenization, stemming)
    ↓
Feature Extraction Module (TF-IDF vectorization)
    ↓
Classification Module (ML prediction)
    ↓
User Feedback (optional correction)
    ↓
Retraining Module (model improvement)
    ↓
Updated Model
```

## 🎮 Usage Guide

### GUI Application

1. **Launch the application**
   ```bash
   python spam_detection_system/main.py
   ```

2. **Classify an email**
   - Enter email text in the text area
   - Click "Classify Email"
   - View prediction results and confidence scores

3. **Provide feedback**
   - After classification, click "Correct" or "Incorrect"
   - The system will use your feedback to improve future predictions

4. **Manage the model**
   - Use "Retrain Model" to update the model with new feedback
   - Save/load models using the model management buttons
   - View system statistics in the statistics panel

### Command Line Usage

```python
from modules.preprocessing import TextPreprocessor
from modules.feature_extraction import FeatureExtractor
from modules.classification import SpamClassifier

# Initialize components
preprocessor = TextPreprocessor()
extractor = FeatureExtractor()
classifier = SpamClassifier(model_type='naive_bayes')

# Train on sample data
emails = ["URGENT! You have won $1000!", "Hi John, thanks for the meeting."]
labels = [1, 0]  # 1 for spam, 0 for ham

processed_emails = preprocessor.preprocess_batch(emails)
X = extractor.extract_features(processed_emails, labels, fit=True)
classifier.train(X, labels)

# Classify new email
new_email = "Win free money now!"
processed = preprocessor.preprocess(new_email)
X_new = extractor.extract_features([processed], fit=False)
prediction = classifier.predict(X_new)
print(f"Prediction: {'Spam' if prediction[0] == 1 else 'Ham'}")
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python tests/test_modules.py
```

The test suite includes:
- Unit tests for all modules
- Integration tests for the complete pipeline
- Performance tests for model accuracy
- Feedback system tests

## 📊 Performance Metrics

The system is designed to achieve:
- **Accuracy**: ≥ 90% on validation datasets
- **False Positives**: < 5%
- **False Negatives**: < 5%
- **Processing Speed**: Up to 100 emails per minute
- **Preprocessing Time**: < 5 seconds per email

## 🔄 Adaptive Learning

The system continuously improves through:

1. **User Feedback Collection**: Users can correct misclassifications
2. **Automatic Retraining**: Model retrains when sufficient feedback is collected
3. **Performance Monitoring**: Tracks accuracy improvements over time
4. **Feature Importance**: Identifies important features for spam detection

## 🛠️ Customization

### Adding New Features

1. **Custom Preprocessing**: Modify `TextPreprocessor` class
2. **New ML Models**: Add to `SpamClassifier` class
3. **Feature Engineering**: Extend `FeatureExtractor` class
4. **UI Components**: Modify GUI in `main.py`

### Configuration

- **Model Parameters**: Adjust in `train_model.py`
- **Feature Extraction**: Configure in `FeatureExtractor` initialization
- **Retraining Thresholds**: Modify in `feedback_retraining.py`

## 📈 Future Enhancements

- **Real-time Email Scanning**: API integration for live email processing
- **Phishing Detection**: Advanced NLP techniques for phishing-specific patterns
- **Multi-language Support**: Extend to other languages
- **Cloud Deployment**: Deploy as a web service
- **Advanced UI**: Web-based interface with modern design

## 🐛 Troubleshooting

### Common Issues

1. **NLTK Data Missing**
   ```python
   import nltk
   nltk.download('all')
   ```

2. **spaCy Model Not Found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Memory Issues with Large Datasets**
   - Reduce `max_features` in `FeatureExtractor`
   - Use smaller `ngram_range`
   - Process data in batches

4. **Low Accuracy**
   - Increase training data
   - Tune hyperparameters
   - Try different preprocessing settings
   - Collect more user feedback

### Performance Optimization

- Use `LinearSVC` instead of `SVC` for faster training
- Reduce feature dimensions with feature selection
- Use sparse matrices for memory efficiency
- Implement batch processing for large datasets

## 📝 License

This project is developed for educational purposes as part of a computer science course assignment.

## 🤝 Contributing

This is a course project, but suggestions and improvements are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📞 Support

For questions or issues:
- Check the troubleshooting section
- Review the test cases for usage examples
- Examine the module documentation

---

**Note**: This system is designed for educational purposes and should be thoroughly tested before use in production environments.

#   s p a m - d e t e c t i o n - s y s t e m  
 