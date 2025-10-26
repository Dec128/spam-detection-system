"""
Text Preprocessing Module for Spam Detection System
==================================================

This module provides comprehensive text preprocessing capabilities for email spam detection.
It handles all aspects of text cleaning, normalization, and preparation for machine learning.

Key Features:
- Text cleaning and normalization (URLs, emails, phone numbers, special characters)
- Advanced tokenization using NLTK
- Stop word removal with custom spam-specific stop words
- Stemming and lemmatization (NLTK and spaCy support)
- Batch processing capabilities
- Statistical analysis of preprocessing results

Classes:
    TextPreprocessor: Main preprocessing class with comprehensive text processing pipeline

Author: Spam Detection System Team
Version: 2.0
Date: 2025
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
import spacy

class TextPreprocessor:
    """
    Comprehensive text preprocessing class for email spam detection.
    
    This class provides a complete pipeline for preprocessing email text, including
    cleaning, tokenization, stop word removal, and stemming/lemmatization. It supports
    both NLTK and spaCy for advanced NLP operations and includes spam-specific optimizations.
    
    Attributes:
        use_lemmatization (bool): Whether to use lemmatization (True) or stemming (False)
        language (str): Language for stop words and processing
        stemmer (PorterStemmer): NLTK stemmer for word stemming
        lemmatizer (WordNetLemmatizer): NLTK lemmatizer for word lemmatization
        stop_words (set): Set of stop words to remove
        nlp (spacy.Language): spaCy language model for advanced processing
        use_spacy (bool): Whether spaCy is available and being used
    """
    
    def __init__(self, use_lemmatization=True, language='english'):
        """
        Initialize the preprocessor
        
        Args:
            use_lemmatization (bool): Whether to use lemmatization (True) or stemming (False)
            language (str): Language for stop words and processing
        """
        self.use_lemmatization = use_lemmatization
        self.language = language
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize tools
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words(language))
        
        # Try to load spaCy model for better lemmatization
        try:
            self.nlp = spacy.load('en_core_web_sm')
            self.use_spacy = True
        except OSError:
            print("Warning: spaCy model 'en_core_web_sm' not found. Using NLTK lemmatization.")
            self.use_spacy = False
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
    
    def _get_wordnet_pos(self, word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    
    def clean_text(self, text):
        """
        Clean the input text by removing special characters and normalizing
        
        Args:
            text (str): Raw email text
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        
        # Remove phone numbers
        text = re.sub(r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}', ' ', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize_text(self, text):
        """
        Tokenize the cleaned text
        
        Args:
            text (str): Cleaned text
            
        Returns:
            list: List of tokens
        """
        if not text:
            return []
        
        try:
            tokens = word_tokenize(text)
            return tokens
        except Exception as e:
            print(f"Error in tokenization: {e}")
            return text.split()
    
    def remove_stop_words(self, tokens):
        """
        Remove stop words from tokens
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Tokens with stop words removed
        """
        if not tokens:
            return []
        
        # Add custom stop words for spam detection
        custom_stop_words = {
            'subject', 're', 'fw', 'fwd', 'urgent', 'important', 'free', 'win', 'winner',
            'congratulations', 'click', 'here', 'now', 'today', 'limited', 'offer',
            'guaranteed', 'risk', 'money', 'cash', 'prize', 'award', 'winner'
        }
        
        all_stop_words = self.stop_words.union(custom_stop_words)
        
        filtered_tokens = [token for token in tokens if token not in all_stop_words]
        return filtered_tokens
    
    def stem_or_lemmatize(self, tokens):
        """
        Apply stemming or lemmatization to tokens
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Processed tokens
        """
        if not tokens:
            return []
        
        if self.use_lemmatization:
            if self.use_spacy:
                # Use spaCy for better lemmatization
                text = ' '.join(tokens)
                doc = self.nlp(text)
                return [token.lemma_ for token in doc if token.lemma_.isalpha()]
            else:
                # Use NLTK lemmatization
                return [self.lemmatizer.lemmatize(token, self._get_wordnet_pos(token)) 
                       for token in tokens if token.isalpha()]
        else:
            # Use stemming
            return [self.stemmer.stem(token) for token in tokens if token.isalpha()]
    
    def preprocess(self, text):
        """
        Complete preprocessing pipeline
        
        Args:
            text (str): Raw email text
            
        Returns:
            list: Preprocessed tokens ready for feature extraction
        """
        if not text:
            return []
        
        # Step 1: Clean text
        cleaned_text = self.clean_text(text)
        
        # Step 2: Tokenize
        tokens = self.tokenize_text(cleaned_text)
        
        # Step 3: Remove stop words
        filtered_tokens = self.remove_stop_words(tokens)
        
        # Step 4: Stem or lemmatize
        processed_tokens = self.stem_or_lemmatize(filtered_tokens)
        
        # Step 5: Filter out very short tokens
        final_tokens = [token for token in processed_tokens if len(token) > 2]
        
        return final_tokens
    
    def preprocess_batch(self, texts):
        """
        Preprocess a batch of texts
        
        Args:
            texts (list): List of raw email texts
            
        Returns:
            list: List of preprocessed token lists
        """
        return [self.preprocess(text) for text in texts]
    
    def get_preprocessing_stats(self, texts):
        """
        Get statistics about the preprocessing results
        
        Args:
            texts (list): List of raw email texts
            
        Returns:
            dict: Statistics about preprocessing
        """
        processed_texts = self.preprocess_batch(texts)
        
        stats = {
            'total_emails': len(texts),
            'avg_tokens_per_email': sum(len(tokens) for tokens in processed_texts) / len(texts) if texts else 0,
            'total_unique_tokens': len(set(token for tokens in processed_texts for token in tokens)),
            'empty_emails': sum(1 for tokens in processed_texts if not tokens)
        }
        
        return stats


def test_preprocessor():
    """Test function for the preprocessor"""
    # Sample spam and ham emails
    sample_emails = [
        "URGENT! You have won $1000! Click here now to claim your prize!",
        "Hi John, thanks for the meeting yesterday. Let's schedule a follow-up.",
        "FREE MONEY! Guaranteed cash prize! No risk! Click now!",
        "Meeting reminder: Project review at 3 PM today in conference room A."
    ]
    
    preprocessor = TextPreprocessor()
    
    print("Testing Text Preprocessor:")
    print("=" * 50)
    
    for i, email in enumerate(sample_emails, 1):
        print(f"\nEmail {i}: {email}")
        processed = preprocessor.preprocess(email)
        print(f"Processed: {processed}")
    
    # Get statistics
    stats = preprocessor.get_preprocessing_stats(sample_emails)
    print(f"\nPreprocessing Statistics:")
    print(f"Total emails: {stats['total_emails']}")
    print(f"Average tokens per email: {stats['avg_tokens_per_email']:.2f}")
    print(f"Total unique tokens: {stats['total_unique_tokens']}")
    print(f"Empty emails: {stats['empty_emails']}")


if __name__ == "__main__":
    test_preprocessor()
