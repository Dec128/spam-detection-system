"""
Comprehensive Test Suite for Spam Detection System
=================================================

This module provides comprehensive unit tests for all components of the spam detection system.
It tests the complete pipeline from text preprocessing to adaptive learning, ensuring
all modules function correctly and meet performance requirements.

Test Coverage:
- TextPreprocessor: Text cleaning, tokenization, stop word removal, stemming/lemmatization
- FeatureExtractor: TF-IDF vectorization, feature selection, statistical analysis
- SpamClassifier: Multiple ML algorithms, training, prediction, model persistence
- FeedbackCollector: User feedback collection, statistics, data management
- ModelRetrainer: Model retraining logic, performance improvement tracking
- AdaptiveLearningSystem: Complete adaptive learning pipeline integration

Test Features:
- Unit tests for individual components
- Integration tests for complete workflows
- Performance validation tests
- Error handling and edge case testing
- Model persistence and loading tests
- Cross-validation and metrics testing

Author: Spam Detection System Team
Version: 2.0
Date: 2025
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd

# Add modules to path for proper imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import all modules to be tested
from modules.preprocessing import TextPreprocessor
from modules.feature_extraction import FeatureExtractor
from modules.classification import SpamClassifier
from modules.feedback_retraining import FeedbackCollector, ModelRetrainer, AdaptiveLearningSystem

class TestTextPreprocessor(unittest.TestCase):
    """
    Test cases for TextPreprocessor module.
    
    This test class validates the text preprocessing pipeline including:
    - Text cleaning and normalization
    - Tokenization and stop word removal
    - Stemming and lemmatization
    - Batch processing capabilities
    - Statistical analysis of preprocessing results
    """
    
    def setUp(self):
        """
        Set up test fixtures before each test method.
        
        Initializes a TextPreprocessor instance and sample email data
        for testing various preprocessing functionalities.
        """
        self.preprocessor = TextPreprocessor()
        self.sample_emails = [
            "URGENT! You have won $1000! Click here now!",
            "Hi John, thanks for the meeting yesterday.",
            "FREE MONEY! Guaranteed cash prize!",
            "Meeting reminder: Project review at 3 PM."
        ]
    
    def test_preprocessing_pipeline(self):
        """
        Test the complete preprocessing pipeline.
        
        Validates that the preprocessing pipeline correctly processes email text
        and returns a list of string tokens. Ensures all tokens are strings
        and the pipeline handles various email formats.
        """
        for email in self.sample_emails:
            processed = self.preprocessor.preprocess(email)
            self.assertIsInstance(processed, list)
            self.assertTrue(all(isinstance(token, str) for token in processed))
    
    def test_batch_preprocessing(self):
        """
        Test batch preprocessing functionality.
        
        Validates that batch preprocessing correctly processes multiple emails
        and returns the expected number of processed email lists.
        """
        processed_emails = self.preprocessor.preprocess_batch(self.sample_emails)
        self.assertEqual(len(processed_emails), len(self.sample_emails))
        self.assertTrue(all(isinstance(tokens, list) for tokens in processed_emails))
    
    def test_text_cleaning(self):
        """
        Test text cleaning functionality.
        
        Validates that the text cleaning process correctly removes:
        - Special characters and punctuation
        - Phone numbers and monetary amounts
        - URLs and email addresses
        """
        dirty_text = "URGENT! You have won $1000! Click here now! Call 555-1234"
        cleaned = self.preprocessor.clean_text(dirty_text)
        self.assertNotIn("$1000", cleaned)
        self.assertNotIn("555-1234", cleaned)
        self.assertNotIn("!", cleaned)
    
    def test_stop_word_removal(self):
        """
        Test stop word removal functionality.
        
        Validates that common stop words are correctly removed while
        meaningful words are preserved in the token list.
        """
        tokens = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
        filtered = self.preprocessor.remove_stop_words(tokens)
        self.assertNotIn("the", filtered)
        self.assertNotIn("over", filtered)
        self.assertIn("quick", filtered)
    
    def test_preprocessing_stats(self):
        """
        Test preprocessing statistics calculation.
        
        Validates that preprocessing statistics are correctly calculated
        and contain all expected metrics for analysis.
        """
        stats = self.preprocessor.get_preprocessing_stats(self.sample_emails)
        self.assertIn('total_emails', stats)
        self.assertIn('avg_tokens_per_email', stats)
        self.assertIn('total_unique_tokens', stats)
        self.assertIn('empty_emails', stats)


class TestFeatureExtractor(unittest.TestCase):
    """
    Test cases for FeatureExtractor module.
    
    This test class validates the feature extraction pipeline including:
    - TF-IDF vectorization and feature extraction
    - Bag-of-Words count vectorization
    - Feature selection using statistical tests
    - Feature name extraction and statistics
    - Dimensionality and sparsity analysis
    """
    
    def setUp(self):
        """
        Set up test fixtures before each test method.
        
        Initializes a FeatureExtractor instance with sample email data
        and preprocessed tokens for testing feature extraction functionalities.
        """
        self.extractor = FeatureExtractor(max_features=100)
        self.sample_emails = [
            "URGENT! You have won $1000! Click here now!",
            "Hi John, thanks for the meeting yesterday.",
            "FREE MONEY! Guaranteed cash prize!",
            "Meeting reminder: Project review at 3 PM."
        ]
        self.labels = [1, 0, 1, 0]
        
        # Preprocess emails for feature extraction
        self.preprocessor = TextPreprocessor()
        self.processed_emails = self.preprocessor.preprocess_batch(self.sample_emails)
    
    def test_tfidf_extraction(self):
        """
        Test TF-IDF feature extraction.
        
        Validates that TF-IDF vectorization correctly extracts features
        with the expected dimensions and non-zero feature count.
        """
        X = self.extractor.extract_tfidf_features(self.processed_emails, fit=True)
        self.assertEqual(X.shape[0], len(self.sample_emails))
        self.assertGreater(X.shape[1], 0)
    
    def test_count_extraction(self):
        """
        Test Bag-of-Words count feature extraction.
        
        Validates that count vectorization correctly extracts features
        with the expected dimensions and non-zero feature count.
        """
        X = self.extractor.extract_count_features(self.processed_emails, fit=True)
        self.assertEqual(X.shape[0], len(self.sample_emails))
        self.assertGreater(X.shape[1], 0)
    
    def test_feature_selection(self):
        """
        Test feature selection using statistical tests.
        
        Validates that feature selection correctly reduces dimensionality
        while maintaining the same number of documents.
        """
        X = self.extractor.extract_tfidf_features(self.processed_emails, fit=True)
        X_selected = self.extractor.select_features(X, self.labels, k=20)
        self.assertEqual(X_selected.shape[0], X.shape[0])
        self.assertEqual(X_selected.shape[1], 20)
    
    def test_feature_names(self):
        """
        Test feature name extraction.
        
        Validates that feature names are correctly extracted and returned
        as a list with non-zero length.
        """
        X = self.extractor.extract_tfidf_features(self.processed_emails, fit=True)
        feature_names = self.extractor.get_feature_names()
        self.assertIsInstance(feature_names, list)
        self.assertGreater(len(feature_names), 0)
    
    def test_feature_stats(self):
        """
        Test feature statistics calculation.
        
        Validates that feature statistics are correctly calculated
        and contain all expected metrics for analysis.
        """
        stats = self.extractor.get_feature_stats(self.processed_emails)
        self.assertIn('num_documents', stats)
        self.assertIn('num_features', stats)
        self.assertIn('sparsity', stats)


class TestSpamClassifier(unittest.TestCase):
    """
    Test cases for SpamClassifier module.
    
    This test class validates the machine learning classification pipeline including:
    - Multiple ML algorithm training and evaluation
    - Prediction and probability calculation
    - Cross-validation and performance metrics
    - Model persistence and loading
    - Hyperparameter optimization
    """
    
    def setUp(self):
        """
        Set up test fixtures before each test method.
        
        Initializes sample email data, preprocesses text, extracts features,
        and prepares training data for classifier testing.
        """
        self.sample_emails = [
            "URGENT! You have won $1000! Click here now!",
            "Hi John, thanks for the meeting yesterday.",
            "FREE MONEY! Guaranteed cash prize!",
            "Meeting reminder: Project review at 3 PM.",
            "Congratulations! You are the winner!",
            "Please find attached the quarterly report."
        ]
        self.labels = [1, 0, 1, 0, 1, 0]
        
        # Preprocess and extract features for training
        self.preprocessor = TextPreprocessor()
        self.extractor = FeatureExtractor(max_features=50)
        self.processed_emails = self.preprocessor.preprocess_batch(self.sample_emails)
        self.X = self.extractor.extract_features(self.processed_emails, self.labels, fit=True)
    
    def test_naive_bayes_classifier(self):
        """
        Test Naive Bayes classifier training and evaluation.
        
        Validates that the Naive Bayes classifier can be trained successfully,
        achieves reasonable accuracy, and provides comprehensive metrics.
        """
        classifier = SpamClassifier(model_type='naive_bayes')
        metrics = classifier.train(self.X, self.labels)
        
        self.assertTrue(classifier.is_trained)
        self.assertIn('accuracy', metrics)
        self.assertIn('f1_score', metrics)
        self.assertGreater(metrics['accuracy'], 0)
    
    def test_svm_classifier(self):
        """
        Test Support Vector Machine classifier training and evaluation.
        
        Validates that the SVM classifier can be trained successfully
        and achieves reasonable performance on the test data.
        """
        classifier = SpamClassifier(model_type='svm')
        metrics = classifier.train(self.X, self.labels)
        
        self.assertTrue(classifier.is_trained)
        self.assertIn('accuracy', metrics)
        self.assertGreater(metrics['accuracy'], 0)
    
    def test_prediction(self):
        """
        Test prediction functionality.
        
        Validates that trained classifiers can make predictions
        with the correct format and valid label values (0 or 1).
        """
        classifier = SpamClassifier(model_type='naive_bayes')
        classifier.train(self.X, self.labels)
        
        predictions = classifier.predict(self.X)
        self.assertEqual(len(predictions), len(self.labels))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
    
    def test_prediction_probabilities(self):
        """
        Test prediction probability calculation.
        
        Validates that probability predictions are correctly calculated
        with proper dimensions and probability constraints (sum to 1.0).
        """
        classifier = SpamClassifier(model_type='naive_bayes')
        classifier.train(self.X, self.labels)
        
        probabilities = classifier.predict_proba(self.X)
        self.assertEqual(probabilities.shape[0], len(self.labels))
        self.assertEqual(probabilities.shape[1], 2)
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))
    
    def test_cross_validation(self):
        """
        Test cross-validation functionality.
        
        Validates that cross-validation correctly evaluates model performance
        and returns expected metrics and score arrays.
        """
        classifier = SpamClassifier(model_type='naive_bayes')
        cv_results = classifier.cross_validate(self.X, self.labels, cv=3)
        
        self.assertIn('cv_scores', cv_results)
        self.assertIn('mean_cv_score', cv_results)
        self.assertEqual(len(cv_results['cv_scores']), 3)
    
    def test_model_save_load(self):
        """
        Test model persistence (saving and loading).
        
        Validates that trained models can be saved to disk and loaded
        correctly while maintaining all model properties and state.
        """
        classifier = SpamClassifier(model_type='naive_bayes')
        classifier.train(self.X, self.labels)
        
        # Save model to temporary file
        test_file = 'test_model.pkl'
        classifier.save_model(test_file)
        self.assertTrue(os.path.exists(test_file))
        
        # Load model and verify properties
        new_classifier = SpamClassifier()
        new_classifier.load_model(test_file)
        
        self.assertTrue(new_classifier.is_trained)
        self.assertEqual(new_classifier.model_type, 'naive_bayes')
        
        # Clean up temporary file
        os.remove(test_file)


class TestFeedbackCollector(unittest.TestCase):
    """
    Test cases for FeedbackCollector module.
    
    This test class validates the user feedback collection system including:
    - Feedback data collection and storage
    - Feedback statistics calculation
    - Incorrect prediction identification
    - Data formatting for retraining
    - JSON persistence and loading
    """
    
    def setUp(self):
        """
        Set up test fixtures before each test method.
        
        Initializes a FeedbackCollector instance with a test JSON file
        for isolated testing of feedback collection functionality.
        """
        self.collector = FeedbackCollector('test_feedback.json')
    
    def tearDown(self):
        """
        Clean up test fixtures after each test method.
        
        Removes the test feedback JSON file to ensure clean test isolation.
        """
        if os.path.exists('test_feedback.json'):
            os.remove('test_feedback.json')
    
    def test_add_feedback(self):
        """
        Test adding user feedback to the collection system.
        
        Validates that feedback is correctly stored with all metadata
        including email text, predicted label, user correction, and accuracy flag.
        """
        self.collector.add_feedback(
            "Test email", 1, 0, 0.8, "user1"
        )
        
        self.assertEqual(len(self.collector.feedback_data), 1)
        self.assertEqual(self.collector.feedback_data[0]['email_text'], "Test email")
        self.assertEqual(self.collector.feedback_data[0]['predicted_label'], 1)
        self.assertEqual(self.collector.feedback_data[0]['user_correction'], 0)
        self.assertFalse(self.collector.feedback_data[0]['is_correct'])
    
    def test_feedback_stats(self):
        """
        Test feedback statistics calculation.
        
        Validates that feedback statistics are correctly calculated
        including accuracy, false positives, false negatives, and total counts.
        """
        # Add mixed feedback (correct and incorrect predictions)
        self.collector.add_feedback("Email 1", 1, 1, 0.9)  # Correct
        self.collector.add_feedback("Email 2", 1, 0, 0.8)  # Incorrect
        self.collector.add_feedback("Email 3", 0, 0, 0.7)  # Correct
        
        stats = self.collector.get_feedback_stats()
        self.assertEqual(stats['total_feedback'], 3)
        self.assertEqual(stats['correct_predictions'], 2)
        self.assertEqual(stats['false_positives'], 1)
        self.assertEqual(stats['false_negatives'], 0)
        self.assertAlmostEqual(stats['accuracy'], 2/3, places=2)
    
    def test_get_incorrect_predictions(self):
        """
        Test retrieval of incorrect predictions.
        
        Validates that the system correctly identifies and returns
        only the feedback entries where predictions were incorrect.
        """
        self.collector.add_feedback("Email 1", 1, 1, 0.9)  # Correct
        self.collector.add_feedback("Email 2", 1, 0, 0.8)  # Incorrect
        self.collector.add_feedback("Email 3", 0, 1, 0.7)  # Incorrect
        
        incorrect = self.collector.get_incorrect_predictions()
        self.assertEqual(len(incorrect), 2)
    
    def test_get_feedback_for_retraining(self):
        """
        Test formatting feedback data for model retraining.
        
        Validates that feedback data is correctly formatted for retraining
        with proper minimum feedback requirements and data structure.
        """
        # Add sufficient feedback for retraining
        for i in range(15):
            self.collector.add_feedback(f"Email {i}", 1, 0, 0.8)
        
        texts, labels = self.collector.get_feedback_for_retraining(min_feedback=10)
        self.assertIsNotNone(texts)
        self.assertIsNotNone(labels)
        self.assertEqual(len(texts), 15)
        self.assertEqual(len(labels), 15)


class TestModelRetrainer(unittest.TestCase):
    """
    Test cases for ModelRetrainer module.
    
    This test class validates the model retraining system including:
    - Retraining decision logic based on feedback and accuracy
    - Model retraining with feedback data
    - Performance improvement tracking
    - Retraining result validation
    """
    
    def setUp(self):
        """
        Set up test fixtures before each test method.
        
        Initializes all components needed for model retraining testing
        and adds sufficient feedback data for retraining validation.
        """
        self.preprocessor = TextPreprocessor()
        self.extractor = FeatureExtractor(max_features=50)
        self.classifier = SpamClassifier(model_type='naive_bayes')
        self.collector = FeedbackCollector('test_feedback.json')
        self.retrainer = ModelRetrainer(
            self.preprocessor, self.extractor, self.classifier, self.collector
        )
        
        # Add sufficient feedback for retraining tests
        for i in range(25):
            self.collector.add_feedback(f"Test email {i}", 1, 0, 0.8)
    
    def tearDown(self):
        """
        Clean up test fixtures after each test method.
        
        Removes the test feedback JSON file to ensure clean test isolation.
        """
        if os.path.exists('test_feedback.json'):
            os.remove('test_feedback.json')
    
    def test_should_retrain(self):
        """
        Test retraining decision logic.
        
        Validates that the retraining decision logic correctly determines
        when retraining is needed based on feedback count and accuracy thresholds.
        """
        should_retrain = self.retrainer.should_retrain(min_feedback=20, accuracy_threshold=0.8)
        self.assertTrue(should_retrain)
    
    def test_retrain_model(self):
        """
        Test model retraining functionality.
        
        Validates that model retraining completes successfully and returns
        comprehensive results including improvement metrics and success status.
        """
        result = self.retrainer.retrain_model()
        
        self.assertTrue(result['success'])
        self.assertIn('feedback_count', result)
        self.assertIn('improvement', result)
        self.assertIn('model_updated', result)


class TestAdaptiveLearningSystem(unittest.TestCase):
    """
    Test cases for AdaptiveLearningSystem module.
    
    This test class validates the complete adaptive learning system including:
    - Prediction with feedback preparation
    - Feedback submission and processing
    - System status monitoring
    - Integration of all adaptive learning components
    """
    
    def setUp(self):
        """
        Set up test fixtures before each test method.
        
        Initializes the complete adaptive learning system with trained model
        and original training data for comprehensive testing.
        """
        self.preprocessor = TextPreprocessor()
        self.extractor = FeatureExtractor(max_features=50)
        self.classifier = SpamClassifier(model_type='naive_bayes')
        self.adaptive_system = AdaptiveLearningSystem(
            self.preprocessor, self.extractor, self.classifier, 'test_feedback.json'
        )
        
        # Set up training data for the adaptive system
        sample_emails = [
            "URGENT! You have won $1000!",
            "Hi John, thanks for the meeting.",
            "FREE MONEY! Guaranteed cash prize!",
            "Meeting reminder: Project review at 3 PM."
        ]
        labels = [1, 0, 1, 0]
        
        processed_emails = self.preprocessor.preprocess_batch(sample_emails)
        X = self.extractor.extract_features(processed_emails, labels, fit=True)
        self.classifier.train(X, labels)
        self.adaptive_system.set_original_training_data(X, labels)
    
    def tearDown(self):
        """
        Clean up test fixtures after each test method.
        
        Removes the test feedback JSON file to ensure clean test isolation.
        """
        if os.path.exists('test_feedback.json'):
            os.remove('test_feedback.json')
    
    def test_predict_with_feedback(self):
        """
        Test prediction with feedback preparation.
        
        Validates that the adaptive system can make predictions and prepare
        the result for potential user feedback with all required metadata.
        """
        result = self.adaptive_system.predict_with_feedback("Test email")
        
        self.assertIn('prediction', result)
        self.assertIn('confidence', result)
        self.assertIn('probabilities', result)
        self.assertIn('ready_for_feedback', result)
        self.assertTrue(result['ready_for_feedback'])
    
    def test_submit_feedback(self):
        """
        Test feedback submission to the adaptive learning system.
        
        Validates that user feedback is correctly submitted and processed
        with proper status tracking and statistics updates.
        """
        result = self.adaptive_system.submit_feedback(
            "Test email", 1, 0, 0.8, "user1"
        )
        
        self.assertTrue(result['feedback_submitted'])
        self.assertIn('feedback_stats', result)
    
    def test_get_system_status(self):
        """
        Test system status retrieval.
        
        Validates that the system status contains all required information
        including model info, feedback statistics, and retraining history.
        """
        status = self.adaptive_system.get_system_status()
        
        self.assertIn('model_info', status)
        self.assertIn('feedback_stats', status)
        self.assertIn('retraining_history', status)


def run_tests():
    """
    Run all test cases for the Spam Detection System.
    
    This function creates a comprehensive test suite that includes all test classes
    and runs them with detailed output. It provides a unified way to execute
    all tests and returns the overall success status.
    
    Returns:
        bool: True if all tests passed, False if any tests failed
    """
    # Create comprehensive test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes to the suite
    test_classes = [
        TestTextPreprocessor,
        TestFeatureExtractor,
        TestSpamClassifier,
        TestFeedbackCollector,
        TestModelRetrainer,
        TestAdaptiveLearningSystem
    ]
    
    # Load tests from each test class
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    """
    Main execution block for running the test suite.
    
    When this module is run directly, it executes all tests and provides
    clear feedback on the test results with appropriate exit codes.
    """
    success = run_tests()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)

