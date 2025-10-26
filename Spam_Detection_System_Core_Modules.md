# Spam Detection System - Core Modules & Code Documentation

## Project Overview

This document presents the core modules and essential code from the Spam Detection System, a comprehensive machine learning-based email classification system. The system uses Python, Scikit-learn, and NLP libraries to accurately classify emails as spam or normal (ham) with adaptive learning capabilities.

## Core Modules

### 1. Text Preprocessing Module (`modules/preprocessing.py`)

**Purpose**: Comprehensive text preprocessing for email spam detection

**Key Features**:
- Text cleaning and normalization (URLs, emails, phone numbers, special characters)
- Advanced tokenization using NLTK
- Stop word removal with custom spam-specific stop words
- Stemming and lemmatization (NLTK and spaCy support)
- Batch processing capabilities

**Core Class**: `TextPreprocessor`

```python
class TextPreprocessor:
    """
    Comprehensive text preprocessing class for email spam detection.
    
    This class provides a complete pipeline for preprocessing email text, including
    cleaning, tokenization, stop word removal, and stemming/lemmatization.
    """
    
    def __init__(self, use_lemmatization=True, language='english'):
        """
        Initialize the preprocessor with NLP tools.
        
        Args:
            use_lemmatization (bool): Whether to use lemmatization (True) or stemming (False)
            language (str): Language for stop words and processing
        """
        self.use_lemmatization = use_lemmatization
        self.language = language
        # Initialize NLTK tools for text processing
        self.stemmer = PorterStemmer()  # For word stemming
        self.lemmatizer = WordNetLemmatizer()  # For word lemmatization
        self.stop_words = set(stopwords.words(language))  # Common stop words to remove
    
    def clean_text(self, text):
        """
        Clean the input text by removing special characters and normalizing.
        
        This method removes URLs, email addresses, phone numbers, and special characters
        to prepare text for tokenization and feature extraction.
        
        Args:
            text (str): Raw email text
            
        Returns:
            str: Cleaned text ready for further processing
        """
        text = text.lower()  # Convert to lowercase for consistency
        # Remove email addresses using regex pattern
        text = re.sub(r'\S+@\S+', ' ', text)  # Remove email addresses
        # Remove URLs (http/https) using comprehensive regex
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        # Remove phone numbers (US format)
        text = re.sub(r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}', ' ', text)
        # Remove all special characters except letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Remove extra whitespace and normalize spacing
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def preprocess(self, text):
        """
        Complete preprocessing pipeline for email text.
        
        This method applies the full preprocessing pipeline:
        1. Clean text (remove special characters, URLs, etc.)
        2. Tokenize (split into individual words)
        3. Remove stop words (common words that don't carry meaning)
        4. Stem/lemmatize (reduce words to root form)
        5. Filter short tokens (remove very short words)
        
        Args:
            text (str): Raw email text
            
        Returns:
            list: Preprocessed tokens ready for feature extraction
        """
        # Step 1: Clean the text
        cleaned_text = self.clean_text(text)
        # Step 2: Tokenize into individual words
        tokens = self.tokenize_text(cleaned_text)
        # Step 3: Remove common stop words
        filtered_tokens = self.remove_stop_words(tokens)
        # Step 4: Apply stemming or lemmatization
        processed_tokens = self.stem_or_lemmatize(filtered_tokens)
        # Step 5: Filter out very short tokens (less than 3 characters)
        final_tokens = [token for token in processed_tokens if len(token) > 2]
        return final_tokens
```

### 2. Feature Extraction Module (`modules/feature_extraction.py`)

**Purpose**: Convert preprocessed text into numerical features for machine learning

**Key Features**:
- TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- Bag-of-Words count vectorization
- N-gram feature extraction (unigrams, bigrams, trigrams)
- Feature selection using statistical tests (chi2, mutual information)
- Dimensionality reduction and feature optimization

**Core Class**: `FeatureExtractor`

```python
class FeatureExtractor:
    """
    Comprehensive feature extraction class for email spam detection.
    
    This class converts preprocessed text into numerical features suitable for
    machine learning algorithms using TF-IDF and Bag-of-Words vectorization.
    """
    
    def __init__(self, max_features=5000, min_df=2, max_df=0.95, ngram_range=(1, 2)):
        """
        Initialize the feature extractor with vectorization parameters.
        
        Args:
            max_features (int): Maximum number of features to extract
            min_df (int or float): Minimum document frequency for features
            max_df (int or float): Maximum document frequency for features
            ngram_range (tuple): Range of n-grams to extract (e.g., (1, 2) for unigrams and bigrams)
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        
        # Initialize TF-IDF vectorizer for text-to-feature conversion
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,  # Limit number of features for efficiency
            min_df=min_df,  # Ignore terms that appear in fewer than min_df documents
            max_df=max_df,  # Ignore terms that appear in more than max_df documents
            ngram_range=ngram_range,  # Extract unigrams and bigrams
            stop_words=None,  # We handle stop words in preprocessing
            lowercase=False,  # We handle case in preprocessing
            token_pattern=r'\b\w+\b'  # Match word boundaries for tokenization
        )
    
    def extract_features(self, token_lists, labels=None, feature_type='tfidf', 
                        select_features=False, k=1000, fit=True):
        """
        Main method to extract features with optional feature selection.
        
        This method provides a unified interface for feature extraction with
        support for different vectorization methods and feature selection.
        
        Args:
            token_lists (list): List of preprocessed token lists
            labels (array-like, optional): Target labels for feature selection
            feature_type (str): Type of features ('tfidf' or 'count')
            select_features (bool): Whether to apply feature selection
            k (int): Number of features to select
            fit (bool): Whether to fit the vectorizer (True for training, False for prediction)
            
        Returns:
            scipy.sparse.csr_matrix: Feature matrix ready for machine learning
        """
        # Extract base features using specified method
        if feature_type == 'tfidf':
            X = self.extract_tfidf_features(token_lists, fit=fit)
        else:
            X = self.extract_count_features(token_lists, fit=fit)
        
        # Apply feature selection if requested and labels are provided
        if select_features and labels is not None and fit:
            X = self.select_features(X, labels, k=k)
        
        return X
    
    def extract_tfidf_features(self, token_lists, fit=True):
        """
        Extract TF-IDF features from preprocessed tokens.
        
        TF-IDF (Term Frequency-Inverse Document Frequency) gives higher weights
        to terms that are frequent in a document but rare across the corpus.
        
        Args:
            token_lists (list): List of preprocessed token lists
            fit (bool): Whether to fit the vectorizer (True for training, False for prediction)
            
        Returns:
            scipy.sparse.csr_matrix: TF-IDF feature matrix
        """
        # Convert token lists back to text strings for vectorization
        texts = [' '.join(tokens) if tokens else '' for tokens in token_lists]
        
        if fit:
            # Fit the vectorizer on training data and transform
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            self.is_fitted = True
        else:
            # Transform new data using already fitted vectorizer
            if not self.is_fitted:
                raise ValueError("Vectorizer must be fitted before transforming new data")
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        
        return tfidf_matrix
```

### 3. Classification Module (`modules/classification.py`)

**Purpose**: Machine learning classification for spam detection

**Key Features**:
- Multiple ML algorithms (Naive Bayes, SVM, Random Forest, Logistic Regression)
- Hyperparameter optimization using GridSearchCV
- Cross-validation and performance evaluation
- Model ensemble capabilities
- Comprehensive metrics calculation
- Model persistence and loading

**Core Class**: `SpamClassifier`

```python
class SpamClassifier:
    """
    Comprehensive spam classification class with multiple ML algorithms.
    
    This class provides a unified interface for various machine learning algorithms
    used in spam detection, including training, prediction, evaluation, and model persistence.
    """
    
    def __init__(self, model_type='naive_bayes', random_state=42):
        """
        Initialize the spam classifier with specified algorithm.
        
        Args:
            model_type (str): Type of ML algorithm ('naive_bayes', 'svm', 'random_forest', 'logistic_regression')
            random_state (int): Random state for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.is_trained = False
        self.training_history = []  # Store training session history
        self._initialize_model()
    
    def _initialize_model(self):
        """
        Initialize the selected machine learning model.
        
        This method creates the appropriate scikit-learn model instance
        based on the specified model type.
        """
        if self.model_type == 'naive_bayes':
            # Naive Bayes: Fast, good for text classification
            self.model = MultinomialNB(alpha=1.0)  # Laplace smoothing
        elif self.model_type == 'svm':
            # Support Vector Machine: Good for high-dimensional data
            self.model = SVC(kernel='linear', probability=True, random_state=self.random_state)
        elif self.model_type == 'random_forest':
            # Random Forest: Ensemble method, handles overfitting well
            self.model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        elif self.model_type == 'logistic_regression':
            # Logistic Regression: Linear model with probability outputs
            self.model = LogisticRegression(random_state=self.random_state, max_iter=1000)
    
    def train(self, X, y, validation_split=0.2, optimize_hyperparameters=False):
        """
        Train the spam classifier on provided data.
        
        This method handles the complete training pipeline including data splitting,
        hyperparameter optimization, model training, and performance evaluation.
        
        Args:
            X (scipy.sparse.csr_matrix or np.ndarray): Feature matrix
            y (array-like): Target labels (0 for ham, 1 for spam)
            validation_split (float): Fraction of data to use for validation
            optimize_hyperparameters (bool): Whether to optimize hyperparameters
            
        Returns:
            dict: Training results and performance metrics
        """
        # Split data for validation if requested
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=self.random_state, stratify=y
            )
        else:
            # Use all data for training when validation_split = 0
            X_train, X_val, y_train, y_val = X, X, y, y
        
        # Optimize hyperparameters if requested (GridSearchCV)
        if optimize_hyperparameters:
            self._optimize_hyperparameters(X_train, y_train)
        
        # Train the model and measure training time
        start_time = datetime.now()
        self.model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate model performance on validation set
        y_pred = self.model.predict(X_val)
        y_pred_proba = self._get_prediction_probabilities(X_val)
        
        # Calculate comprehensive performance metrics
        metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
        metrics['training_time'] = training_time
        
        # Store training session in history
        training_record = {
            'timestamp': datetime.now(),
            'model_type': self.model_type,
            'training_samples': X_train.shape[0],
            'validation_samples': X_val.shape[0],
            'metrics': metrics
        }
        self.training_history.append(training_record)
        
        self.is_trained = True
        return metrics
    
    def predict(self, X):
        """
        Predict spam/ham labels for new emails.
        
        Args:
            X (scipy.sparse.csr_matrix or np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Predicted labels (0 for ham, 1 for spam)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict spam/ham probabilities for new emails.
        
        This method returns probability scores for each class, which are useful
        for understanding model confidence and making threshold-based decisions.
        
        Args:
            X (scipy.sparse.csr_matrix or np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Prediction probabilities [ham_prob, spam_prob]
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self._get_prediction_probabilities(X)
```

### 4. Feedback & Retraining Module (`modules/feedback_retraining.py`)

**Purpose**: Adaptive learning system with user feedback integration

**Key Features**:
- User feedback collection and storage
- Automatic model retraining based on feedback
- Performance monitoring and trend analysis
- Adaptive learning system integration
- Feedback statistics and analytics

**Core Classes**: `FeedbackCollector`, `ModelRetrainer`, `AdaptiveLearningSystem`

```python
class FeedbackCollector:
    """
    Collects and manages user feedback for model improvement.
    
    This class handles the collection, storage, and analysis of user feedback
    on spam detection predictions, providing data for model retraining.
    """
    
    def __init__(self, feedback_file='data/user_feedback.json'):
        """
        Initialize the feedback collector with JSON storage.
        
        Args:
            feedback_file (str): Path to JSON file storing feedback data
        """
        self.feedback_file = feedback_file
        self.feedback_data = []  # In-memory storage for feedback entries
        self.load_feedback()  # Load existing feedback from file
    
    def add_feedback(self, email_text, predicted_label, user_correction, confidence_score=None, user_id=None):
        """
        Add user feedback for a prediction to improve model accuracy.
        
        This method stores user corrections to model predictions, which are used
        for adaptive learning and model retraining.
        
        Args:
            email_text (str): Original email text that was classified
            predicted_label (int): Model's prediction (0 for ham, 1 for spam)
            user_correction (int): User's correction (0 for ham, 1 for spam)
            confidence_score (float): Model's confidence score for the prediction
            user_id (str): Optional user identifier for tracking
        """
        # Create comprehensive feedback entry with metadata
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),  # When feedback was given
            'email_text': email_text,  # Original email content
            'predicted_label': int(predicted_label),  # Model's prediction
            'user_correction': int(user_correction),  # User's correction
            'confidence_score': confidence_score,  # Model confidence
            'user_id': user_id,  # User identifier
            'is_correct': predicted_label == user_correction  # Whether model was correct
        }
        self.feedback_data.append(feedback_entry)
        self.save_feedback()  # Persist feedback to file

class AdaptiveLearningSystem:
    """
    Complete adaptive learning system that combines feedback collection and retraining.
    
    This class provides a unified interface for the adaptive learning system,
    combining feedback collection, model retraining, and performance monitoring.
    """
    
    def __init__(self, preprocessor, feature_extractor, classifier, feedback_file='data/user_feedback.json'):
        """
        Initialize the adaptive learning system with all required components.
        
        Args:
            preprocessor: TextPreprocessor instance for text processing
            feature_extractor: FeatureExtractor instance for feature extraction
            classifier: SpamClassifier instance for classification
            feedback_file (str): Path to store feedback data
        """
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.feedback_collector = FeedbackCollector(feedback_file)
        self.retrainer = ModelRetrainer(
            preprocessor, feature_extractor, classifier, self.feedback_collector
        )
        self.original_training_data = None  # Store original training data for retraining
    
    def predict_with_feedback(self, email_text, user_id=None):
        """
        Make a prediction and prepare the result for potential user feedback.
        
        This method processes an email through the complete pipeline and returns
        prediction results along with metadata needed for feedback collection.
        
        Args:
            email_text (str): Email text to classify
            user_id (str): Optional user identifier
            
        Returns:
            dict: Prediction result with feedback preparation metadata
        """
        # Process email through the complete pipeline
        processed_email = self.preprocessor.preprocess(email_text)
        X = self.feature_extractor.extract_features([processed_email], fit=False)
        
        # Get model prediction and confidence scores
        prediction = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]
        confidence = max(probabilities)  # Maximum probability across classes
        
        return {
            'email_text': email_text,  # Original email for feedback
            'prediction': int(prediction),  # Model's prediction (0=ham, 1=spam)
            'confidence': float(confidence),  # Model's confidence score
            'probabilities': {
                'ham': float(probabilities[0]),  # Probability of being ham
                'spam': float(probabilities[1])  # Probability of being spam
            },
            'user_id': user_id,  # User identifier
            'ready_for_feedback': True  # Flag indicating feedback can be submitted
        }
    
    def submit_feedback(self, email_text, predicted_label, user_correction, confidence_score=None, user_id=None):
        """
        Submit user feedback and potentially trigger model retraining.
        
        This method processes user feedback, determines if retraining is needed,
        and automatically retrains the model if sufficient feedback is collected.
        
        Args:
            email_text (str): Original email text
            predicted_label (int): Model's prediction
            user_correction (int): User's correction
            confidence_score (float): Model's confidence score
            user_id (str): Optional user identifier
            
        Returns:
            dict: Feedback submission result with retraining status
        """
        # Add feedback to the collection system
        self.feedback_collector.add_feedback(
            email_text, predicted_label, user_correction, confidence_score, user_id
        )
        
        # Check if retraining is needed based on feedback quantity and accuracy
        should_retrain = self.retrainer.should_retrain()
        
        # Prepare result with feedback status
        result = {
            'feedback_submitted': True,  # Confirmation that feedback was stored
            'should_retrain': should_retrain,  # Whether retraining is recommended
            'feedback_stats': self.feedback_collector.get_feedback_stats()  # Current feedback statistics
        }
        
        # Automatically retrain model if conditions are met
        if should_retrain:
            retrain_result = self.retrainer.retrain_model(self.original_training_data)
            result['retraining_result'] = retrain_result
        
        return result
```

## Main Application (`spam_detection_system/main.py`)

**Purpose**: GUI application for the spam detection system

**Key Features**:
- Modern GUI interface with intuitive controls
- Real-time email classification
- User feedback collection
- Model management and retraining
- Status monitoring and statistics

**Core Class**: `SpamDetectionGUI`

```python
class SpamDetectionGUI:
    """
    Main GUI application class for the Spam Detection System.
    
    This class provides a comprehensive user interface for email spam detection,
    including real-time classification, user feedback collection, and model management.
    """
    
    def __init__(self, root):
        """
        Initialize the Spam Detection GUI application.
        
        Args:
            root (tk.Tk): Main Tkinter window instance
        """
        self.root = root
        self.root.title("🛡️ Spam Detection System")
        self.root.geometry("900x800")
        
        # Initialize core components for the ML pipeline
        self.preprocessor = TextPreprocessor()  # Text preprocessing component
        self.feature_extractor = FeatureExtractor(
            max_features=1000,  # Limit features for efficiency
            min_df=1,  # Minimum document frequency
            max_df=0.8,  # Maximum document frequency
            ngram_range=(1, 2)  # Unigrams and bigrams
        )
        self.classifier = None  # Will be initialized when model is loaded/trained
        self.adaptive_system = None  # Adaptive learning system
        self.is_model_trained = False  # Model training status
        self.current_prediction = None  # Current prediction for feedback
        
        # Setup the GUI interface
        self.create_widgets()
        self.setup_layout()
        self.initialize_model()  # Load or create the ML model
    
    def classify_email(self):
        """
        Classify the email text as spam or ham using the trained model.
        
        This method handles the complete classification pipeline:
        1. Validates model readiness
        2. Checks for valid email input
        3. Uses adaptive learning system for prediction
        4. Updates GUI with results and confidence scores
        5. Applies color coding (red for spam, green for ham)
        6. Stores prediction data for potential user feedback
        """
        # Validate that model is ready for classification
        if not self.is_model_trained:
            messagebox.showerror("Error", "Model is not trained.")
            return
        
        # Get email text from the input area
        email_text = self.email_text.get('1.0', tk.END).strip()
        
        # Check for valid email input
        if not email_text or email_text == "Paste your email content here for analysis...":
            messagebox.showwarning("Warning", "Please enter email text to analyze.")
            return
        
        try:
            # Update status and show processing indicator
            self.status_var.set("🔍 Analyzing email...")
            self.root.update()  # Force GUI update
            
            # Use adaptive system for prediction with feedback preparation
            result = self.adaptive_system.predict_with_feedback(email_text)
            
            # Format prediction results for display
            prediction = "🚨 SPAM" if result['prediction'] == 1 else "✅ HAM"
            confidence = result['confidence']
            
            # Update GUI with prediction results
            self.prediction_label.config(text=f"Status: {prediction}")
            self.confidence_label.config(text=f"Confidence: {confidence:.1%}")
            
            # Apply color coding based on prediction
            if result['prediction'] == 1:  # Spam prediction
                self.prediction_label.config(foreground='#e74c3c')  # Red color
                self.confidence_label.config(foreground='#e74c3c')
            else:  # Ham prediction
                self.prediction_label.config(foreground='#27ae60')  # Green color
                self.confidence_label.config(foreground='#27ae60')
            
            # Store prediction for potential user feedback
            self.current_prediction = result
            self.status_var.set("✅ Analysis completed")
            
        except Exception as e:
            # Handle classification errors gracefully
            messagebox.showerror("Error", f"Classification failed: {str(e)}")
            self.status_var.set("Classification failed")
    
    def submit_feedback(self, is_correct):
        """
        Submit user feedback on the current prediction to improve model accuracy.
        
        This method processes user feedback and triggers model retraining if needed.
        The feedback is used to improve the model through adaptive learning.
        
        Args:
            is_correct (bool): True if the prediction was correct, False otherwise
        """
        # Validate that there is a current prediction to provide feedback for
        if not self.current_prediction:
            messagebox.showwarning("Warning", "No prediction to provide feedback for.")
            return
        
        try:
            # Calculate user correction based on feedback
            # If user says prediction is correct, use the same label
            # If user says prediction is incorrect, flip the label
            user_correction = self.current_prediction['prediction'] if is_correct else (1 - self.current_prediction['prediction'])
            
            # Submit feedback to the adaptive learning system
            result = self.adaptive_system.submit_feedback(
                self.current_prediction['email_text'],  # Original email text
                self.current_prediction['prediction'],  # Model's prediction
                user_correction,  # User's correction
                self.current_prediction['confidence']  # Model's confidence
            )
            
            # Handle feedback submission result
            if result['feedback_submitted']:
                messagebox.showinfo("Success", "Feedback submitted successfully!")
                self.save_model()  # Auto-save model after feedback
            else:
                messagebox.showerror("Error", "Failed to submit feedback.")
            
        except Exception as e:
            # Handle feedback submission errors
            messagebox.showerror("Error", f"Failed to submit feedback: {str(e)}")
```

## Training Dataset Module (`data/training_dataset.py`)

**Purpose**: Dataset management and SpamAssassin Public Corpus integration

**Key Features**:
- SpamAssassin Public Corpus support
- Email parsing and content extraction
- Dataset statistics and information
- Fallback sample data

```python
def load_spamassassin_data(dataset_path: str, max_emails_per_category: int = 100) -> Tuple[List[str], List[int]]:
    """
    Load SpamAssassin Public Corpus dataset for training the spam detection model.
    
    This function loads the widely-used SpamAssassin Public Corpus, which contains
    real-world email data with known spam/ham labels. The dataset is organized into
    three categories: easy_ham (normal emails), hard_ham (difficult normal emails),
    and spam (spam emails).
    
    Args:
        dataset_path (str): Path to SpamAssassin dataset directory
        max_emails_per_category (int): Maximum emails per category to load (for memory efficiency)
        
    Returns:
        Tuple[List[str], List[int]]: (email contents, labels) where labels are 0 for ham, 1 for spam
    """
    emails = []  # Store email content
    labels = []    # Store corresponding labels (0=ham, 1=spam)
    
    # Define dataset categories and their corresponding labels
    categories = [
        ('easy_ham', 0),    # Normal emails that are easy to classify
        ('hard_ham', 0),    # Normal emails that are difficult to classify (edge cases)
        ('spam', 1)         # Spam emails
    ]
    
    # Process each category in the dataset
    for category, label in categories:
        category_path = os.path.join(dataset_path, category)
        
        # Check if category directory exists
        if not os.path.exists(category_path):
            print(f"Warning: Directory {category_path} does not exist, skipping...")
            continue
            
        # Get all email files in this category
        email_files = glob.glob(os.path.join(category_path, '*'))
        email_files = [f for f in email_files if os.path.isfile(f)]  # Filter out directories
        
        # Limit the number of emails per category to manage memory usage
        if max_emails_per_category and len(email_files) > max_emails_per_category:
            email_files = email_files[:max_emails_per_category]
        
        print(f"Loading {category}: {len(email_files)} emails")
        
        # Process each email file in the category
        for email_file in email_files:
            try:
                # Load and parse the email file
                email_content = _load_email_file(email_file)
                if email_content:  # Only add if email was successfully parsed
                    emails.append(email_content)
                    labels.append(label)
            except Exception as e:
                # Handle individual email parsing errors gracefully
                print(f"Failed to load email {email_file}: {e}")
                continue  # Skip this email and continue with the next
    
    return emails, labels
```

## Usage Example

Here's a complete example demonstrating how to use the Spam Detection System:

```python
# Initialize the complete spam detection system
from modules.preprocessing import TextPreprocessor
from modules.feature_extraction import FeatureExtractor
from modules.classification import SpamClassifier
from modules.feedback_retraining import AdaptiveLearningSystem

# Create system components for the ML pipeline
preprocessor = TextPreprocessor()  # Text preprocessing component
extractor = FeatureExtractor(max_features=1000)  # Feature extraction with 1000 features
classifier = SpamClassifier(model_type='naive_bayes')  # Naive Bayes classifier
adaptive_system = AdaptiveLearningSystem(preprocessor, extractor, classifier)  # Complete adaptive system

# Train the system with sample data
sample_emails = ["URGENT! You have won $1000!", "Hi John, thanks for the meeting."]
labels = [1, 0]  # 1 for spam, 0 for ham (normal email)

# Process emails through the complete pipeline
processed_emails = preprocessor.preprocess_batch(sample_emails)  # Clean and tokenize emails
X = extractor.extract_features(processed_emails, labels, fit=True)  # Extract numerical features
classifier.train(X, labels)  # Train the machine learning model
adaptive_system.set_original_training_data(X, labels)  # Store training data for retraining

# Classify a new email using the trained system
result = adaptive_system.predict_with_feedback("Win free money now!")
print(f"Prediction: {'Spam' if result['prediction'] == 1 else 'Ham'}")
print(f"Confidence: {result['confidence']:.2%}")

# Submit user feedback to improve the model
# In this example, user corrects the spam prediction to ham (0)
feedback_result = adaptive_system.submit_feedback(
    "Win free money now!",  # Original email text
    result['prediction'],   # Model's prediction (1 for spam)
    0,                      # User's correction (0 for ham)
    result['confidence']    # Model's confidence score
)
print(f"Feedback submitted: {feedback_result['feedback_submitted']}")
print(f"Should retrain: {feedback_result['should_retrain']}")
```

## Key Features Summary

1. **Advanced Text Processing**: NLTK and spaCy integration for sophisticated text preprocessing
2. **Multiple ML Algorithms**: Support for Naive Bayes, SVM, Random Forest, and Logistic Regression
3. **Feature Engineering**: TF-IDF and Bag-of-Words with feature selection
4. **Adaptive Learning**: User feedback integration with automatic model retraining
5. **Comprehensive Testing**: Full test suite with 665 lines of test code
6. **Modern GUI**: Intuitive interface with real-time classification
7. **Model Persistence**: Save/load functionality for trained models
8. **Performance Monitoring**: Statistics and trend analysis

This documentation presents the core modules and essential code from the Spam Detection System, demonstrating a comprehensive machine learning application with advanced features, adaptive learning capabilities, and thorough testing implementation.