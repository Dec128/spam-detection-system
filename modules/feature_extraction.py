"""
Feature Extraction Module for Spam Detection System
=================================================

This module provides comprehensive feature extraction capabilities for email spam detection.
It converts preprocessed text into numerical features suitable for machine learning algorithms.

Key Features:
- TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- Bag-of-Words count vectorization
- N-gram feature extraction (unigrams, bigrams, trigrams)
- Feature selection using statistical tests (chi2, mutual information)
- Dimensionality reduction and feature optimization
- Feature importance analysis and ranking
- Model persistence and loading capabilities

Classes:
    FeatureExtractor: Main feature extraction class with comprehensive vectorization pipeline

Author: Spam Detection System Team
Version: 2.0
Date: 2025
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
import pickle
import os

class FeatureExtractor:
    """
    Comprehensive feature extraction class for email spam detection.
    
    This class provides advanced feature extraction capabilities using TF-IDF and
    Bag-of-Words vectorization, with support for feature selection and optimization.
    It handles both training and prediction phases with proper feature alignment.
    
    Attributes:
        max_features (int): Maximum number of features to extract
        min_df (int or float): Minimum document frequency for features
        max_df (int or float): Maximum document frequency for features
        ngram_range (tuple): Range of n-grams to extract (e.g., (1, 2) for unigrams and bigrams)
        tfidf_vectorizer (TfidfVectorizer): TF-IDF vectorizer instance
        count_vectorizer (CountVectorizer): Count vectorizer instance
        feature_selector (SelectKBest): Feature selector for dimensionality reduction
        selected_features (array): Indices of selected features
        is_fitted (bool): Whether the vectorizer has been fitted to data
    """
    
    def __init__(self, max_features=5000, min_df=2, max_df=0.95, ngram_range=(1, 2)):
        """
        Initialize the feature extractor
        
        Args:
            max_features (int): Maximum number of features to extract
            min_df (int or float): Minimum document frequency for features
            max_df (int or float): Maximum document frequency for features
            ngram_range (tuple): Range of n-grams to extract
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        
        # Initialize vectorizers
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            stop_words=None,  # We handle stop words in preprocessing
            lowercase=False,  # We handle case in preprocessing
            token_pattern=r'\b\w+\b'  # Match word boundaries
        )
        
        self.count_vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            stop_words=None,
            lowercase=False,
            token_pattern=r'\b\w+\b'
        )
        
        # Feature selection
        self.feature_selector = None
        self.selected_features = None
        self.is_fitted = False
    
    def _tokens_to_text(self, token_lists):
        """
        Convert list of token lists to list of text strings for vectorization
        
        Args:
            token_lists (list): List of token lists from preprocessing
            
        Returns:
            list: List of text strings
        """
        return [' '.join(tokens) if tokens else '' for tokens in token_lists]
    
    def extract_tfidf_features(self, token_lists, fit=True):
        """
        Extract TF-IDF features from preprocessed tokens
        
        Args:
            token_lists (list): List of preprocessed token lists
            fit (bool): Whether to fit the vectorizer (True for training, False for prediction)
            
        Returns:
            scipy.sparse.csr_matrix: TF-IDF feature matrix
        """
        # Convert tokens to text
        texts = self._tokens_to_text(token_lists)
        
        if fit:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Vectorizer must be fitted before transforming new data")
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        
        return tfidf_matrix
    
    def extract_count_features(self, token_lists, fit=True):
        """
        Extract Bag-of-Words count features from preprocessed tokens
        
        Args:
            token_lists (list): List of preprocessed token lists
            fit (bool): Whether to fit the vectorizer (True for training, False for prediction)
            
        Returns:
            scipy.sparse.csr_matrix: Count feature matrix
        """
        # Convert tokens to text
        texts = self._tokens_to_text(token_lists)
        
        if fit:
            count_matrix = self.count_vectorizer.fit_transform(texts)
        else:
            if not self.is_fitted:
                raise ValueError("Vectorizer must be fitted before transforming new data")
            count_matrix = self.count_vectorizer.transform(texts)
        
        return count_matrix
    
    def select_features(self, X, y, k=1000, method='chi2'):
        """
        Select the best k features using statistical tests
        
        Args:
            X (scipy.sparse.csr_matrix): Feature matrix
            y (array-like): Target labels
            k (int): Number of features to select
            method (str): Feature selection method ('chi2', 'mutual_info_classif')
            
        Returns:
            scipy.sparse.csr_matrix: Selected feature matrix
        """
        if method == 'chi2':
            self.feature_selector = SelectKBest(chi2, k=k)
        else:
            from sklearn.feature_selection import mutual_info_classif
            self.feature_selector = SelectKBest(mutual_info_classif, k=k)
        
        X_selected = self.feature_selector.fit_transform(X, y)
        self.selected_features = self.feature_selector.get_support(indices=True)
        
        return X_selected
    
    def extract_features(self, token_lists, labels=None, feature_type='tfidf', 
                        select_features=False, k=1000, fit=True):
        """
        Main method to extract features with optional feature selection
        
        Args:
            token_lists (list): List of preprocessed token lists
            labels (array-like, optional): Target labels for feature selection
            feature_type (str): Type of features ('tfidf' or 'count')
            select_features (bool): Whether to apply feature selection
            k (int): Number of features to select
            fit (bool): Whether to fit the vectorizer
            
        Returns:
            scipy.sparse.csr_matrix: Feature matrix
        """
        # Extract base features
        if feature_type == 'tfidf':
            X = self.extract_tfidf_features(token_lists, fit=fit)
        else:
            X = self.extract_count_features(token_lists, fit=fit)
        
        # Apply feature selection if requested
        if select_features and labels is not None and fit:
            X = self.select_features(X, labels, k=k)
        
        return X
    
    def get_feature_names(self):
        """
        Get the feature names from the vectorizer
        
        Returns:
            list: List of feature names
        """
        if self.is_fitted:
            if hasattr(self, 'feature_selector') and self.feature_selector is not None:
                # Return selected feature names
                all_features = self.tfidf_vectorizer.get_feature_names_out()
                return [all_features[i] for i in self.selected_features]
            else:
                return self.tfidf_vectorizer.get_feature_names_out().tolist()
        else:
            return []
    
    def get_feature_importance(self, X, y, top_n=20):
        """
        Get feature importance scores
        
        Args:
            X (scipy.sparse.csr_matrix): Feature matrix
            y (array-like): Target labels
            top_n (int): Number of top features to return
            
        Returns:
            dict: Dictionary with feature names and importance scores
        """
        if not self.is_fitted:
            return {}
        
        # Use chi2 for feature importance
        chi2_scores, _ = chi2(X, y)
        feature_names = self.get_feature_names()
        
        # Create importance dictionary
        importance_dict = dict(zip(feature_names, chi2_scores))
        
        # Sort by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_features[:top_n])
    
    def save_vectorizer(self, filepath):
        """
        Save the fitted vectorizer to disk
        
        Args:
            filepath (str): Path to save the vectorizer
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before saving")
        
        vectorizer_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'count_vectorizer': self.count_vectorizer,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features,
            'is_fitted': self.is_fitted,
            'max_features': self.max_features,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'ngram_range': self.ngram_range
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(vectorizer_data, f)
    
    def load_vectorizer(self, filepath):
        """
        Load a fitted vectorizer from disk
        
        Args:
            filepath (str): Path to load the vectorizer from
        """
        with open(filepath, 'rb') as f:
            vectorizer_data = pickle.load(f)
        
        self.tfidf_vectorizer = vectorizer_data['tfidf_vectorizer']
        self.count_vectorizer = vectorizer_data['count_vectorizer']
        self.feature_selector = vectorizer_data['feature_selector']
        self.selected_features = vectorizer_data['selected_features']
        self.is_fitted = vectorizer_data['is_fitted']
        self.max_features = vectorizer_data['max_features']
        self.min_df = vectorizer_data['min_df']
        self.max_df = vectorizer_data['max_df']
        self.ngram_range = vectorizer_data['ngram_range']
    
    def get_feature_stats(self, token_lists):
        """
        Get statistics about the extracted features
        
        Args:
            token_lists (list): List of preprocessed token lists
            
        Returns:
            dict: Statistics about features
        """
        if not token_lists:
            return {}
        
        # Extract features
        X = self.extract_features(token_lists, fit=False)
        
        stats = {
            'num_documents': len(token_lists),
            'num_features': X.shape[1],
            'sparsity': 1.0 - (X.nnz / (X.shape[0] * X.shape[1])),
            'avg_features_per_doc': X.sum(axis=1).mean(),
            'max_features_per_doc': X.sum(axis=1).max(),
            'min_features_per_doc': X.sum(axis=1).min()
        }
        
        return stats


def test_feature_extractor():
    """Test function for the feature extractor"""
    from modules.preprocessing import TextPreprocessor
    
    # Sample data
    sample_emails = [
        "URGENT! You have won $1000! Click here now to claim your prize!",
        "Hi John, thanks for the meeting yesterday. Let's schedule a follow-up.",
        "FREE MONEY! Guaranteed cash prize! No risk! Click now!",
        "Meeting reminder: Project review at 3 PM today in conference room A.",
        "Congratulations! You are the winner of our lottery! Claim now!",
        "Please find attached the quarterly report for your review."
    ]
    
    labels = [1, 0, 1, 0, 1, 0]  # 1 for spam, 0 for ham
    
    # Preprocess emails
    preprocessor = TextPreprocessor()
    processed_emails = preprocessor.preprocess_batch(sample_emails)
    
    print("Testing Feature Extractor:")
    print("=" * 50)
    
    # Test TF-IDF features
    extractor = FeatureExtractor(max_features=100, ngram_range=(1, 2))
    X_tfidf = extractor.extract_features(processed_emails, labels, feature_type='tfidf', fit=True)
    
    print(f"TF-IDF Features Shape: {X_tfidf.shape}")
    print(f"Feature Names (first 10): {extractor.get_feature_names()[:10]}")
    
    # Test feature selection
    X_selected = extractor.extract_features(processed_emails, labels, feature_type='tfidf', 
                                          select_features=True, k=20, fit=True)
    print(f"Selected Features Shape: {X_selected.shape}")
    
    # Get feature importance
    importance = extractor.get_feature_importance(X_tfidf, labels, top_n=10)
    print(f"Top 10 Most Important Features:")
    for feature, score in importance.items():
        print(f"  {feature}: {score:.4f}")
    
    # Get statistics
    stats = extractor.get_feature_stats(processed_emails)
    print(f"\nFeature Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_feature_extractor()
