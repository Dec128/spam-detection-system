"""
Classification Module for Spam Detection System
==============================================

This module provides comprehensive machine learning classification capabilities for
email spam detection. It supports multiple algorithms and includes advanced features
like hyperparameter optimization, cross-validation, and model ensemble methods.

Key Features:
- Multiple ML algorithms (Naive Bayes, SVM, Random Forest, Logistic Regression)
- Hyperparameter optimization using GridSearchCV
- Cross-validation and performance evaluation
- Model ensemble capabilities
- Comprehensive metrics calculation
- Model persistence and loading
- Feature importance analysis

Classes:
    SpamClassifier: Main classification class with multiple ML algorithms
    ModelEnsemble: Ensemble classifier combining multiple models

Author: Spam Detection System Team
Version: 2.0
Date: 2025
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pickle
import os
from datetime import datetime

class SpamClassifier:
    """
    Comprehensive spam classification class with multiple ML algorithms.
    
    This class provides a unified interface for various machine learning algorithms
    used in spam detection. It supports training, prediction, evaluation, and model
    persistence with consistent APIs across different algorithms.
    
    Attributes:
        model_type (str): Type of ML algorithm ('naive_bayes', 'svm', 'random_forest', etc.)
        random_state (int): Random state for reproducibility
        model: The actual scikit-learn model instance
        is_trained (bool): Whether the model has been trained
        training_history (list): History of training sessions with metrics
    """
    
    def __init__(self, model_type='naive_bayes', random_state=42):
        """
        Initialize the spam classifier
        
        Args:
            model_type (str): Type of model to use ('naive_bayes', 'svm', 'random_forest', 'logistic_regression')
            random_state (int): Random state for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.is_trained = False
        self.training_history = []
        
        # Initialize the model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the selected model"""
        if self.model_type == 'naive_bayes':
            self.model = MultinomialNB(alpha=1.0)
        elif self.model_type == 'gaussian_naive_bayes':
            self.model = GaussianNB()
        elif self.model_type == 'svm':
            self.model = SVC(kernel='linear', probability=True, random_state=self.random_state)
        elif self.model_type == 'linear_svm':
            self.model = LinearSVC(random_state=self.random_state, max_iter=10000)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X, y, validation_split=0.2, optimize_hyperparameters=False):
        """
        Train the spam classifier
        
        Args:
            X (scipy.sparse.csr_matrix or np.ndarray): Feature matrix
            y (array-like): Target labels (0 for ham, 1 for spam)
            validation_split (float): Fraction of data to use for validation
            optimize_hyperparameters (bool): Whether to optimize hyperparameters
            
        Returns:
            dict: Training results and metrics
        """
        # Split data for validation (if validation_split > 0)
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=self.random_state, stratify=y
            )
        else:
            # Use all data for training when validation_split = 0
            X_train, X_val, y_train, y_val = X, X, y, y
        
        # Optimize hyperparameters if requested
        if optimize_hyperparameters:
            self._optimize_hyperparameters(X_train, y_train)
        
        # Train the model
        start_time = datetime.now()
        self.model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate on validation set
        y_pred = self.model.predict(X_val)
        y_pred_proba = self._get_prediction_probabilities(X_val)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
        metrics['training_time'] = training_time
        
        # Store training history
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
    
    def _optimize_hyperparameters(self, X, y):
        """Optimize hyperparameters using GridSearchCV"""
        param_grids = {
            'naive_bayes': {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]},
            'gaussian_naive_bayes': {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]},
            'svm': {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']},
            'linear_svm': {'C': [0.1, 1, 10, 100]},
            'random_forest': {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]},
            'logistic_regression': {'C': [0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
        }
        
        if self.model_type in param_grids:
            param_grid = param_grids[self.model_type]
            grid_search = GridSearchCV(
                self.model, param_grid, cv=3, scoring='f1', n_jobs=-1
            )
            grid_search.fit(X, y)
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    def predict(self, X):
        """
        Predict spam/ham labels for new emails
        
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
        Predict spam/ham probabilities for new emails
        
        Args:
            X (scipy.sparse.csr_matrix or np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Prediction probabilities [ham_prob, spam_prob]
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self._get_prediction_probabilities(X)
    
    def _get_prediction_probabilities(self, X):
        """Get prediction probabilities, handling different model types"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'decision_function'):
            # For LinearSVC, use decision function to approximate probabilities
            decision_scores = self.model.decision_function(X)
            # Convert to probabilities using sigmoid
            probabilities = 1 / (1 + np.exp(-decision_scores))
            return np.column_stack([1 - probabilities, probabilities])
        else:
            # Fallback: return predictions as probabilities
            predictions = self.model.predict(X)
            return np.column_stack([1 - predictions, predictions])
    
    def evaluate(self, X, y):
        """
        Evaluate the model on test data
        
        Args:
            X (scipy.sparse.csr_matrix or np.ndarray): Feature matrix
            y (array-like): True labels
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)
        
        return self._calculate_metrics(y, y_pred, y_pred_proba)
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'precision_spam': precision_score(y_true, y_pred, pos_label=1),
            'recall_spam': recall_score(y_true, y_pred, pos_label=1),
            'f1_spam': f1_score(y_true, y_pred, pos_label=1),
            'precision_ham': precision_score(y_true, y_pred, pos_label=0),
            'recall_ham': recall_score(y_true, y_pred, pos_label=0),
            'f1_ham': f1_score(y_true, y_pred, pos_label=0)
        }
        
        # Add confidence scores
        if y_pred_proba is not None:
            max_proba = np.max(y_pred_proba, axis=1)
            metrics['avg_confidence'] = np.mean(max_proba)
            metrics['min_confidence'] = np.min(max_proba)
            metrics['max_confidence'] = np.max(max_proba)
        
        return metrics
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation
        
        Args:
            X (scipy.sparse.csr_matrix or np.ndarray): Feature matrix
            y (array-like): Target labels
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Cross-validation results
        """
        if not self.is_trained:
            # Train the model first
            self.model.fit(X, y)
            self.is_trained = True
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='f1')
        
        return {
            'cv_scores': cv_scores,
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores),
            'cv_folds': cv
        }
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance (for models that support it)
        
        Args:
            feature_names (list): List of feature names
            
        Returns:
            dict: Feature importance scores
        """
        if not self.is_trained:
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            return {}
        
        if feature_names is not None and len(feature_names) == len(importances):
            return dict(zip(feature_names, importances))
        else:
            return {f'feature_{i}': importance for i, importance in enumerate(importances)}
    
    def save_model(self, filepath):
        """
        Save the trained model to disk
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'random_state': self.random_state
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """
        Load a trained model from disk
        
        Args:
            filepath (str): Path to load the model from
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        self.training_history = model_data['training_history']
        self.random_state = model_data['random_state']
    
    def get_model_info(self):
        """Get information about the current model"""
        return {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'training_sessions': len(self.training_history),
            'last_training': self.training_history[-1] if self.training_history else None
        }


class ModelEnsemble:
    """
    Ensemble classifier combining multiple models
    """
    
    def __init__(self, models=None, voting='soft'):
        """
        Initialize ensemble classifier
        
        Args:
            models (list): List of SpamClassifier instances
            voting (str): Voting method ('hard' or 'soft')
        """
        self.models = models or []
        self.voting = voting
        self.is_trained = False
    
    def add_model(self, model):
        """Add a model to the ensemble"""
        self.models.append(model)
    
    def train(self, X, y, validation_split=0.2):
        """Train all models in the ensemble"""
        for model in self.models:
            model.train(X, y, validation_split=validation_split)
        self.is_trained = True
    
    def predict(self, X):
        """Make ensemble predictions"""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Majority voting
        predictions = np.array(predictions)
        ensemble_pred = np.round(np.mean(predictions, axis=0))
        
        return ensemble_pred.astype(int)
    
    def predict_proba(self, X):
        """Make ensemble probability predictions"""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        probabilities = []
        for model in self.models:
            proba = model.predict_proba(X)
            probabilities.append(proba)
        
        # Average probabilities
        ensemble_proba = np.mean(probabilities, axis=0)
        
        return ensemble_proba


def test_classifier():
    """Test function for the classifier"""
    from modules.preprocessing import TextPreprocessor
    from modules.feature_extraction import FeatureExtractor
    
    # Sample data
    sample_emails = [
        "URGENT! You have won $1000! Click here now to claim your prize!",
        "Hi John, thanks for the meeting yesterday. Let's schedule a follow-up.",
        "FREE MONEY! Guaranteed cash prize! No risk! Click now!",
        "Meeting reminder: Project review at 3 PM today in conference room A.",
        "Congratulations! You are the winner of our lottery! Claim now!",
        "Please find attached the quarterly report for your review.",
        "Limited time offer! Get 50% off now! Don't miss out!",
        "Thanks for your email. I'll get back to you soon."
    ]
    
    labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1 for spam, 0 for ham
    
    # Preprocess emails
    preprocessor = TextPreprocessor()
    processed_emails = preprocessor.preprocess_batch(sample_emails)
    
    # Extract features
    extractor = FeatureExtractor(max_features=100)
    X = extractor.extract_features(processed_emails, labels, fit=True)
    
    print("Testing Spam Classifier:")
    print("=" * 50)
    
    # Test different models
    models_to_test = ['naive_bayes', 'svm', 'random_forest', 'logistic_regression']
    
    for model_type in models_to_test:
        print(f"\nTesting {model_type}:")
        print("-" * 30)
        
        classifier = SpamClassifier(model_type=model_type)
        metrics = classifier.train(X, labels, validation_split=0.25)
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Precision (Spam): {metrics['precision_spam']:.4f}")
        print(f"Recall (Spam): {metrics['recall_spam']:.4f}")
        
        # Test prediction
        test_emails = ["Win free money now!", "Meeting at 3 PM"]
        test_processed = preprocessor.preprocess_batch(test_emails)
        test_X = extractor.extract_features(test_processed, fit=False)
        test_pred = classifier.predict(test_X)
        test_proba = classifier.predict_proba(test_X)
        
        print(f"Test predictions: {test_pred}")
        print(f"Test probabilities: {test_proba}")


if __name__ == "__main__":
    test_classifier()
