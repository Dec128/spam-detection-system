"""
Feedback and Retraining Module for Spam Detection System
========================================================

This module provides comprehensive adaptive learning capabilities for the spam detection
system. It handles user feedback collection, model retraining, and performance monitoring
to continuously improve classification accuracy.

Key Features:
- User feedback collection and storage
- Automatic model retraining based on feedback
- Performance monitoring and trend analysis
- Adaptive learning system integration
- Feedback statistics and analytics
- Model improvement tracking

Classes:
    FeedbackCollector: Collects and manages user feedback data
    ModelRetrainer: Handles model retraining based on feedback
    AdaptiveLearningSystem: Complete adaptive learning system

Author: Spam Detection System Team
Version: 2.0
Date: 2025
"""

import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime
from collections import defaultdict
import json

class FeedbackCollector:
    """
    Collects and manages user feedback for model improvement.
    
    This class handles the collection, storage, and analysis of user feedback
    on spam detection predictions. It provides statistics and data formatting
    for model retraining and performance monitoring.
    
    Attributes:
        feedback_file (str): Path to JSON file storing feedback data
        feedback_data (list): List of feedback entries with timestamps and metadata
    """
    
    def __init__(self, feedback_file='data/user_feedback.json'):
        """
        Initialize the feedback collector
        
        Args:
            feedback_file (str): Path to store feedback data
        """
        self.feedback_file = feedback_file
        self.feedback_data = []
        self.load_feedback()
    
    def add_feedback(self, email_text, predicted_label, user_correction, confidence_score=None, user_id=None):
        """
        Add user feedback for a prediction
        
        Args:
            email_text (str): Original email text
            predicted_label (int): Model's prediction (0 for ham, 1 for spam)
            user_correction (int): User's correction (0 for ham, 1 for spam)
            confidence_score (float): Model's confidence score
            user_id (str): Optional user identifier
        """
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'email_text': email_text,
            'predicted_label': int(predicted_label),
            'user_correction': int(user_correction),
            'confidence_score': confidence_score,
            'user_id': user_id,
            'is_correct': predicted_label == user_correction
        }
        
        self.feedback_data.append(feedback_entry)
        self.save_feedback()
    
    def get_feedback_stats(self):
        """
        Get statistics about collected feedback
        
        Returns:
            dict: Feedback statistics
        """
        if not self.feedback_data:
            return {
                'total_feedback': 0,
                'accuracy': 0.0,
                'false_positives': 0,
                'false_negatives': 0,
                'correct_predictions': 0
            }
        
        total_feedback = len(self.feedback_data)
        correct_predictions = sum(1 for entry in self.feedback_data if entry['is_correct'])
        false_positives = sum(1 for entry in self.feedback_data 
                            if entry['predicted_label'] == 1 and entry['user_correction'] == 0)
        false_negatives = sum(1 for entry in self.feedback_data 
                            if entry['predicted_label'] == 0 and entry['user_correction'] == 1)
        
        return {
            'total_feedback': total_feedback,
            'accuracy': correct_predictions / total_feedback if total_feedback > 0 else 0.0,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'correct_predictions': correct_predictions
        }
    
    def get_incorrect_predictions(self, limit=None):
        """
        Get emails that were incorrectly classified
        
        Args:
            limit (int): Maximum number of incorrect predictions to return
            
        Returns:
            list: List of incorrect prediction entries
        """
        incorrect = [entry for entry in self.feedback_data if not entry['is_correct']]
        return incorrect[:limit] if limit else incorrect
    
    def get_feedback_for_retraining(self, min_feedback=10):
        """
        Get feedback data formatted for retraining
        
        Args:
            min_feedback (int): Minimum number of feedback entries required
            
        Returns:
            tuple: (email_texts, corrected_labels) or (None, None) if insufficient data
        """
        if len(self.feedback_data) < min_feedback:
            return None, None
        
        email_texts = [entry['email_text'] for entry in self.feedback_data]
        corrected_labels = [entry['user_correction'] for entry in self.feedback_data]
        
        return email_texts, corrected_labels
    
    def save_feedback(self):
        """Save feedback data to file"""
        os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
        with open(self.feedback_file, 'w', encoding='utf-8') as f:
            json.dump(self.feedback_data, f, indent=2, ensure_ascii=False)
    
    def load_feedback(self):
        """Load feedback data from file"""
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    self.feedback_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                self.feedback_data = []
        else:
            self.feedback_data = []


class ModelRetrainer:
    """
    Handles model retraining based on user feedback.
    
    This class manages the retraining process using collected user feedback.
    It determines when retraining is needed, combines original and feedback data,
    and evaluates the improvement in model performance.
    
    Attributes:
        preprocessor: TextPreprocessor instance for text processing
        feature_extractor: FeatureExtractor instance for feature extraction
        classifier: SpamClassifier instance for classification
        feedback_collector: FeedbackCollector instance for feedback management
        retraining_history (list): History of retraining sessions with results
    """
    
    def __init__(self, preprocessor, feature_extractor, classifier, feedback_collector):
        """
        Initialize the model retrainer
        
        Args:
            preprocessor: TextPreprocessor instance
            feature_extractor: FeatureExtractor instance
            classifier: SpamClassifier instance
            feedback_collector: FeedbackCollector instance
        """
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.feedback_collector = feedback_collector
        self.retraining_history = []
    
    def should_retrain(self, min_feedback=20, accuracy_threshold=0.8):
        """
        Determine if the model should be retrained based on feedback
        
        Args:
            min_feedback (int): Minimum number of feedback entries required
            accuracy_threshold (float): Accuracy threshold below which retraining is needed
            
        Returns:
            bool: Whether retraining is recommended
        """
        stats = self.feedback_collector.get_feedback_stats()
        
        if stats['total_feedback'] < min_feedback:
            return False
        
        if stats['accuracy'] < accuracy_threshold:
            return True
        
        return False
    
    def retrain_model(self, original_training_data=None, retrain_threshold=0.1):
        """
        Retrain the model using user feedback
        
        Args:
            original_training_data (tuple): Original training data (X, y) if available
            retrain_threshold (float): Minimum improvement threshold for retraining
        
        Returns:
            dict: Retraining results
        """
        # Get feedback data
        feedback_texts, feedback_labels = self.feedback_collector.get_feedback_for_retraining()
        
        if feedback_texts is None:
            return {
                'success': False,
                'message': 'Insufficient feedback data for retraining',
                'feedback_count': len(self.feedback_collector.feedback_data)
            }
        
        # Preprocess feedback data
        feedback_processed = self.preprocessor.preprocess_batch(feedback_texts)
        
        # Extract features for feedback data
        feedback_X = self.feature_extractor.extract_features(
            feedback_processed, feedback_labels, fit=False
        )
        
        # Evaluate current model on feedback data
        current_metrics = self.classifier.evaluate(feedback_X, feedback_labels)
        
        # Create retraining data
        if original_training_data is not None:
            # Combine original and feedback data
            X_original, y_original = original_training_data
            X_combined = np.vstack([X_original.toarray(), feedback_X.toarray()])
            y_combined = np.concatenate([y_original, feedback_labels])
        else:
            # Use only feedback data
            X_combined = feedback_X
            y_combined = np.array(feedback_labels)
        
        # Retrain the model
        retrain_start = datetime.now()
        
        # Create a new classifier instance to avoid modifying the original
        from modules.classification import SpamClassifier
        new_classifier = SpamClassifier(
            model_type=self.classifier.model_type,
            random_state=self.classifier.random_state
        )
        
        # Train the new model
        new_classifier.train(X_combined, y_combined, validation_split=0.2)
        
        # Evaluate the new model
        new_metrics = new_classifier.evaluate(feedback_X, feedback_labels)
        
        retrain_time = (datetime.now() - retrain_start).total_seconds()
        
        # Check if improvement is significant
        improvement = new_metrics['accuracy'] - current_metrics['accuracy']
        
        retrain_result = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'feedback_count': len(feedback_texts),
            'current_accuracy': current_metrics['accuracy'],
            'new_accuracy': new_metrics['accuracy'],
            'improvement': improvement,
            'retrain_time': retrain_time,
            'model_updated': improvement > retrain_threshold
        }
        
        # Update the classifier if improvement is significant
        if improvement > retrain_threshold:
            self.classifier = new_classifier
            retrain_result['message'] = f'Model retrained successfully. Accuracy improved by {improvement:.4f}'
        else:
            retrain_result['message'] = f'Retraining completed but improvement ({improvement:.4f}) below threshold ({retrain_threshold})'
        
        # Store retraining history
        self.retraining_history.append(retrain_result)
        
        return retrain_result
    
    def get_retraining_history(self):
        """Get the retraining history"""
        return self.retraining_history
    
    def get_model_performance_trend(self):
        """
        Get model performance trend over time
        
        Returns:
            dict: Performance trend data
        """
        if not self.retraining_history:
            return {}
        
        timestamps = [entry['timestamp'] for entry in self.retraining_history]
        accuracies = [entry['new_accuracy'] for entry in self.retraining_history]
        improvements = [entry['improvement'] for entry in self.retraining_history]
        
        return {
            'timestamps': timestamps,
            'accuracies': accuracies,
            'improvements': improvements,
            'total_retrains': len(self.retraining_history)
        }


class AdaptiveLearningSystem:
    """
    Complete adaptive learning system that combines feedback collection and retraining.
    
    This class provides a unified interface for the adaptive learning system,
    combining feedback collection, model retraining, and performance monitoring.
    It handles the complete cycle of prediction, feedback collection, and model improvement.
    
    Attributes:
        preprocessor: TextPreprocessor instance for text processing
        feature_extractor: FeatureExtractor instance for feature extraction
        classifier: SpamClassifier instance for classification
        feedback_collector: FeedbackCollector instance for feedback management
        retrainer: ModelRetrainer instance for model retraining
        original_training_data (tuple): Original training data (X, y) for retraining
    """
    
    def __init__(self, preprocessor, feature_extractor, classifier, 
                 feedback_file='data/user_feedback.json'):
        """
        Initialize the adaptive learning system
        
        Args:
            preprocessor: TextPreprocessor instance
            feature_extractor: FeatureExtractor instance
            classifier: SpamClassifier instance
            feedback_file (str): Path to store feedback data
        """
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.feedback_collector = FeedbackCollector(feedback_file)
        self.retrainer = ModelRetrainer(
            preprocessor, feature_extractor, classifier, self.feedback_collector
        )
        self.original_training_data = None
    
    def predict_with_feedback(self, email_text, user_id=None):
        """
        Make a prediction and prepare for potential feedback
        
        Args:
            email_text (str): Email text to classify
            user_id (str): Optional user identifier
            
        Returns:
            dict: Prediction result with feedback preparation
        """
        # Preprocess email
        processed_email = self.preprocessor.preprocess(email_text)
        
        # Extract features
        X = self.feature_extractor.extract_features([processed_email], fit=False)
        
        # Make prediction
        prediction = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]
        confidence = max(probabilities)
        
        return {
            'email_text': email_text,
            'prediction': int(prediction),
            'confidence': float(confidence),
            'probabilities': {
                'ham': float(probabilities[0]),
                'spam': float(probabilities[1])
            },
            'user_id': user_id,
            'ready_for_feedback': True
        }
    
    def submit_feedback(self, email_text, predicted_label, user_correction, 
                       confidence_score=None, user_id=None):
        """
        Submit user feedback and potentially trigger retraining
        
        Args:
            email_text (str): Original email text
            predicted_label (int): Model's prediction
            user_correction (int): User's correction
            confidence_score (float): Model's confidence score
            user_id (str): Optional user identifier
            
        Returns:
            dict: Feedback submission result
        """
        # Add feedback
        self.feedback_collector.add_feedback(
            email_text, predicted_label, user_correction, confidence_score, user_id
        )
        
        # Check if retraining is needed
        should_retrain = self.retrainer.should_retrain()
        
        result = {
            'feedback_submitted': True,
            'should_retrain': should_retrain,
            'feedback_stats': self.feedback_collector.get_feedback_stats()
        }
        
        # Retrain if needed
        if should_retrain:
            retrain_result = self.retrainer.retrain_model(self.original_training_data)
            result['retraining_result'] = retrain_result
        
        return result
    
    def set_original_training_data(self, X, y):
        """
        Set the original training data for retraining
        
        Args:
            X: Original feature matrix
            y: Original labels
        """
        self.original_training_data = (X, y)
    
    def get_system_status(self):
        """
        Get comprehensive system status
        
        Returns:
            dict: System status information
        """
        feedback_stats = self.feedback_collector.get_feedback_stats()
        retraining_history = self.retrainer.get_retraining_history()
        model_info = self.classifier.get_model_info()
        
        return {
            'model_info': model_info,
            'feedback_stats': feedback_stats,
            'retraining_history': retraining_history,
            'total_retrains': len(retraining_history),
            'last_retrain': retraining_history[-1] if retraining_history else None
        }


def test_feedback_system():
    """Test function for the feedback and retraining system"""
    from modules.preprocessing import TextPreprocessor
    from modules.feature_extraction import FeatureExtractor
    from modules.classification import SpamClassifier
    
    # Initialize components
    preprocessor = TextPreprocessor()
    extractor = FeatureExtractor(max_features=100)
    classifier = SpamClassifier(model_type='naive_bayes')
    
    # Sample training data
    sample_emails = [
        "URGENT! You have won $1000! Click here now!",
        "Hi John, thanks for the meeting yesterday.",
        "FREE MONEY! Guaranteed cash prize!",
        "Meeting reminder: Project review at 3 PM.",
        "Congratulations! You are the winner!",
        "Please find attached the quarterly report."
    ]
    
    labels = [1, 0, 1, 0, 1, 0]
    
    # Preprocess and extract features
    processed_emails = preprocessor.preprocess_batch(sample_emails)
    X = extractor.extract_features(processed_emails, labels, fit=True)
    
    # Train classifier
    classifier.train(X, labels)
    
    # Initialize adaptive learning system
    adaptive_system = AdaptiveLearningSystem(preprocessor, extractor, classifier)
    adaptive_system.set_original_training_data(X, labels)
    
    print("Testing Adaptive Learning System:")
    print("=" * 50)
    
    # Test prediction and feedback
    test_email = "Win free money now! Limited time offer!"
    prediction = adaptive_system.predict_with_feedback(test_email)
    
    print(f"Test Email: {test_email}")
    print(f"Prediction: {'Spam' if prediction['prediction'] == 1 else 'Ham'}")
    print(f"Confidence: {prediction['confidence']:.4f}")
    
    # Simulate user feedback (correcting spam to ham)
    feedback_result = adaptive_system.submit_feedback(
        test_email, prediction['prediction'], 0, prediction['confidence']
    )
    
    print(f"\nFeedback submitted: {feedback_result['feedback_submitted']}")
    print(f"Should retrain: {feedback_result['should_retrain']}")
    print(f"Feedback stats: {feedback_result['feedback_stats']}")
    
    # Get system status
    status = adaptive_system.get_system_status()
    print(f"\nSystem Status:")
    print(f"Model trained: {status['model_info']['is_trained']}")
    print(f"Total feedback: {status['feedback_stats']['total_feedback']}")
    print(f"Total retrains: {status['total_retrains']}")


if __name__ == "__main__":
    test_feedback_system()
