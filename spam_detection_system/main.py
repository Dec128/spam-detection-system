"""
Spam Detection System - Main Application
========================================

A comprehensive GUI application for email spam detection using machine learning.
This system provides real-time email classification with user feedback integration
and adaptive learning capabilities.

Features:
- Real-time email spam/ham classification
- User feedback collection for model improvement
- Adaptive learning system with automatic retraining
- Modern GUI interface with intuitive controls
- Support for SpamAssassin Public Corpus dataset
- Model persistence and loading capabilities

Author: Spam Detection System Team
Version: 2.0
Date: 2025
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import os
import sys
import pickle
from datetime import datetime

# Add modules to path for proper imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import core modules
from modules.preprocessing import TextPreprocessor
from modules.feature_extraction import FeatureExtractor
from modules.classification import SpamClassifier
from modules.feedback_retraining import AdaptiveLearningSystem

class SpamDetectionGUI:
    """
    Main GUI application class for the Spam Detection System.
    
    This class provides a comprehensive user interface for email spam detection,
    including real-time classification, user feedback collection, and model management.
    
    Attributes:
        root (tk.Tk): Main Tkinter window
        preprocessor (TextPreprocessor): Text preprocessing component
        feature_extractor (FeatureExtractor): Feature extraction component
        classifier (SpamClassifier): Machine learning classifier
        adaptive_system (AdaptiveLearningSystem): Adaptive learning system
        is_model_trained (bool): Whether the model is trained and ready
        current_prediction (dict): Current prediction result for feedback
    """
    
    def __init__(self, root):
        """
        Initialize the Spam Detection GUI application.
        
        Args:
            root (tk.Tk): Main Tkinter window instance
        """
        # Configure main window
        self.root = root
        self.root.title("🛡️ Spam Detection System")
        self.root.geometry("900x800")
        self.root.configure(bg='#f8f9fa')
        self.root.minsize(800, 700)
        
        # Initialize core components
        self.preprocessor = TextPreprocessor()
        self.feature_extractor = FeatureExtractor(
            max_features=1000, 
            min_df=1, 
            max_df=0.8, 
            ngram_range=(1, 2)
        )
        self.classifier = None
        self.adaptive_system = None
        self.is_model_trained = False
        self.current_prediction = None
        
        # Setup GUI components
        self.create_widgets()
        self.setup_layout()
        
        # Initialize and load model
        self.initialize_model()
    
    def create_widgets(self):
        """
        Create and configure all GUI widgets for the application.
        
        This method sets up the modern interface with styled components including:
        - Header section with title and subtitle
        - Email input area with scrollable text widget
        - Action buttons for classification and clearing
        - Results display with color-coded predictions
        - Feedback section for user corrections
        - Model status and management controls
        - Status bar for system notifications
        """
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure custom button styles
        style.configure('Success.TButton', foreground='#27ae60', font=('Segoe UI', 10, 'bold'))
        style.configure('Danger.TButton', foreground='#e74c3c', font=('Segoe UI', 10, 'bold'))
        style.configure('Accent.TButton', font=('Segoe UI', 11, 'bold'))
        
        # Main frame with better spacing
        self.main_frame = ttk.Frame(self.root, padding="20")
        
        # Header with icon and title
        self.header_frame = ttk.Frame(self.main_frame)
        self.title_label = ttk.Label(self.header_frame, text="Spam Detection System", 
                                    font=('Segoe UI', 20, 'bold'), foreground='#2c3e50')
        self.subtitle_label = ttk.Label(self.header_frame, text="AI-powered email classification", 
                                       font=('Segoe UI', 10), foreground='#7f8c8d')
        
        # Email input with modern styling
        self.input_frame = ttk.LabelFrame(self.main_frame, text="📧 Email Input", padding="15")
        self.email_text = scrolledtext.ScrolledText(self.input_frame, height=6, width=70, wrap=tk.WORD,
                                                   font=('Consolas', 10), bg='#ffffff', fg='#2c3e50')
        self.email_text.insert('1.0', "Paste your email content here for analysis...")
        self.email_text.bind('<FocusIn>', self.clear_placeholder)
        
        # Action buttons with better styling
        self.buttons_frame = ttk.Frame(self.input_frame)
        self.classify_btn = ttk.Button(self.buttons_frame, text="🔍 Analyze Email", 
                                      command=self.classify_email, style='Accent.TButton')
        self.clear_btn = ttk.Button(self.buttons_frame, text="🗑️ Clear", command=self.clear_input)
        
        # Results with color coding
        self.results_frame = ttk.LabelFrame(self.main_frame, text="📊 Analysis Results", padding="15")
        self.prediction_label = ttk.Label(self.results_frame, text="Status: Ready for analysis", 
                                         font=('Segoe UI', 14, 'bold'))
        self.confidence_label = ttk.Label(self.results_frame, text="Confidence: N/A", 
                                         font=('Segoe UI', 12))
        
        # Feedback section as separate frame
        self.feedback_frame = ttk.LabelFrame(self.main_frame, text="💬 Feedback", padding="15")
        self.feedback_label = ttk.Label(self.feedback_frame, text="Was this classification correct?", 
                                       font=('Segoe UI', 11))
        self.feedback_buttons_frame = ttk.Frame(self.feedback_frame)
        self.correct_btn = ttk.Button(self.feedback_buttons_frame, text="✅ Correct", 
                                     command=lambda: self.submit_feedback(True),
                                     style='Success.TButton')
        self.incorrect_btn = ttk.Button(self.feedback_buttons_frame, text="❌ Incorrect", 
                                       command=lambda: self.submit_feedback(False),
                                       style='Danger.TButton')
        
        # Model status with better info
        self.model_frame = ttk.LabelFrame(self.main_frame, text="🤖 Model Status", padding="15")
        self.model_status_label = ttk.Label(self.model_frame, text="Model Status: Initializing...", 
                                           font=('Segoe UI', 11))
        self.retrain_btn = ttk.Button(self.model_frame, text="🔄 Retrain Model", 
                                     command=self.retrain_model)
        
        # Status bar with better styling
        self.status_var = tk.StringVar()
        self.status_var.set("🟢 System Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                                   relief=tk.SUNKEN, anchor=tk.W, font=('Segoe UI', 9))
    
    def setup_layout(self):
        """
        Configure the layout and positioning of all GUI components.
        
        This method arranges all widgets in a logical, user-friendly layout with:
        - Proper spacing and padding between components
        - Responsive design that adapts to window resizing
        - Clear visual hierarchy with grouped related elements
        - Professional appearance with consistent styling
        """
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header section with centered title
        self.header_frame.pack(fill=tk.X, pady=(0, 20))
        self.title_label.pack(anchor=tk.CENTER)
        self.subtitle_label.pack(anchor=tk.CENTER, pady=(5, 0))
        
        # Input section with better spacing
        self.input_frame.pack(fill=tk.X, pady=(0, 15))
        self.email_text.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        self.buttons_frame.pack(fill=tk.X)
        self.classify_btn.pack(side=tk.LEFT, padx=(0, 10), ipadx=10, ipady=5)
        self.clear_btn.pack(side=tk.LEFT, ipadx=10, ipady=5)
        
        # Results section with better organization
        self.results_frame.pack(fill=tk.X, pady=(0, 15))
        self.prediction_label.pack(pady=(0, 8))
        self.confidence_label.pack(pady=(0, 8))
        
        # Feedback section as separate frame
        self.feedback_frame.pack(fill=tk.X, pady=(0, 15))
        self.feedback_label.pack(pady=(0, 12))
        self.feedback_buttons_frame.pack()
        self.correct_btn.pack(side=tk.LEFT, padx=(0, 15), ipadx=12, ipady=6)
        self.incorrect_btn.pack(side=tk.LEFT, ipadx=12, ipady=6)
        
        # Model status section with proper layout
        self.model_frame.pack(fill=tk.X, pady=(0, 20))
        self.model_status_label.pack(side=tk.LEFT, padx=(0, 20), pady=5)
        self.retrain_btn.pack(side=tk.RIGHT, ipadx=8, ipady=3, pady=5)
        
        # Status bar
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def clear_placeholder(self, event):
        """
        Clear placeholder text when user clicks on the email input area.
        
        Args:
            event: Tkinter event object (not used but required for binding)
        """
        if self.email_text.get('1.0', tk.END).strip() == "Paste your email content here for analysis...":
            self.email_text.delete('1.0', tk.END)
    
    def clear_input(self):
        """
        Clear the email input area and reset all result displays.
        
        This method resets the interface to its initial state, clearing:
        - Email input text area
        - Prediction results and confidence scores
        - Current prediction data for feedback
        - Status indicators
        """
        self.email_text.delete('1.0', tk.END)
        self.email_text.insert('1.0', "Paste your email content here for analysis...")
        self.prediction_label.config(text="Status: Ready for analysis")
        self.confidence_label.config(text="Confidence: N/A")
        self.current_prediction = None
        self.status_var.set("🟢 Ready for analysis")
    
    def initialize_model(self):
        """
        Initialize or load the machine learning model for spam detection.
        
        This method attempts to:
        1. Load a pre-trained model from disk if available
        2. Search for models in multiple possible locations (exe and script modes)
        3. Create a new model using SpamAssassin dataset if no model is found
        4. Handle both compiled executable and script execution environments
        
        The method supports multiple model file locations for maximum compatibility.
        """
        try:
            # Get the directory where the executable is located
            if getattr(sys, 'frozen', False):
                # Running as compiled exe
                base_path = sys._MEIPASS
            else:
                # Running as script
                base_path = os.path.dirname(os.path.abspath(__file__))
            
            # Try multiple possible model locations
            possible_model_paths = [
                os.path.join(base_path, 'data', 'spam_model.pkl'),
                os.path.join(base_path, 'spam_model.pkl'),
                os.path.join(os.path.dirname(base_path), 'data', 'spam_model.pkl'),
                os.path.join(os.path.dirname(base_path), 'spam_model.pkl'),
                'data/spam_model.pkl',
                'spam_model.pkl'
            ]
            
            model_found = False
            for model_path in possible_model_paths:
                if os.path.exists(model_path):
                    try:
                        self.load_model(model_path)
                        model_found = True
                        break
                    except Exception as e:
                        print(f"Failed to load model from {model_path}: {e}")
                        continue
            
            if not model_found:
                self.create_model()
        except Exception as e:
            print(f"Model initialization failed: {e}")
            self.create_model()
    
    def create_model(self):
        """
        Create a new machine learning model using SpamAssassin Public Corpus.
        
        This method:
        1. Searches for SpamAssassin dataset in multiple common locations
        2. Falls back to sample data if real dataset is not found
        3. Preprocesses the email data using the text preprocessor
        4. Extracts features using TF-IDF vectorization
        5. Trains a Naive Bayes classifier
        6. Initializes the adaptive learning system
        7. Updates the GUI to reflect model readiness
        
        The method handles both real SpamAssassin data and fallback sample data
        to ensure the system always has a working model.
        """
        # Try to load SpamAssassin Public Corpus
        from data.training_dataset import load_spamassassin_data, get_spamassassin_sample_data
        
        sample_emails = None
        labels = None
        dataset_source = "Unknown"
        
        # Get the directory where the executable is located
        if getattr(sys, 'frozen', False):
            # Running as compiled exe
            base_path = sys._MEIPASS
        else:
            # Running as script
            base_path = os.path.dirname(os.path.abspath(__file__))
        
        # Try to find SpamAssassin dataset in common locations
        possible_paths = [
            os.path.join(base_path, "spamassassin"),
            os.path.join(base_path, "data", "spamassassin"),
            os.path.join(os.path.dirname(base_path), "spamassassin"),
            "spamassassin",
            "data/spamassassin", 
            "../spamassassin",
            "C:/spamassassin",
            "D:/spamassassin"
        ]
        
        for dataset_path in possible_paths:
            if os.path.exists(dataset_path):
                try:
                    print(f"Found SpamAssassin dataset at: {dataset_path}")
                    sample_emails, labels = load_spamassassin_data(dataset_path, max_emails_per_category=100)
                    dataset_source = f"SpamAssassin Public Corpus ({dataset_path})"
                    break
                except Exception as e:
                    print(f"Failed to load from {dataset_path}: {e}")
                    continue
        
        # If no real dataset found, use sample data
        if sample_emails is None:
            print("SpamAssassin Public Corpus not found, using sample data")
            sample_emails, labels = get_spamassassin_sample_data()
            dataset_source = "SpamAssassin-style sample data"
        
        print(f"Dataset Source: {dataset_source}")
        print(f"Total emails: {len(sample_emails)}")
        print(f"Spam emails: {labels.count(1)}")
        print(f"Ham emails: {labels.count(0)}")
        
        # Train model
        processed_emails = self.preprocessor.preprocess_batch(sample_emails)
        X = self.feature_extractor.extract_features(processed_emails, labels, fit=True)
        
        self.classifier = SpamClassifier(model_type='naive_bayes')
        training_metrics = self.classifier.train(X, labels, validation_split=0.0, optimize_hyperparameters=False)
        
        # Initialize adaptive system
        self.adaptive_system = AdaptiveLearningSystem(self.preprocessor, self.feature_extractor, self.classifier)
        self.adaptive_system.set_original_training_data(X, labels)
        
        self.is_model_trained = True
        self.model_status_label.config(text="🤖 Model Status: Ready")
        self.status_var.set("🎉 Model ready for analysis")
        
        print(f"Model Training Completed:")
        print(f"Accuracy: {training_metrics['accuracy']:.4f}")
        print(f"F1 Score: {training_metrics['f1_score']:.4f}")
        print(f"Dataset Source: {dataset_source}")
    
    def load_model(self, file_path):
        """Load model from file"""
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.feature_extractor = model_data['feature_extractor']
        self.preprocessor = model_data['preprocessor']
        self.is_model_trained = model_data['is_model_trained']
        
        # Initialize adaptive system
        self.adaptive_system = AdaptiveLearningSystem(self.preprocessor, self.feature_extractor, self.classifier)
        
        self.model_status_label.config(text="Model Status: Loaded")
        self.status_var.set("Model loaded successfully")
    
    def save_model(self, file_path=None):
        """Save model to file"""
        if not self.is_model_trained:
            raise ValueError("No trained model to save.")
        
        # Determine the correct path for saving
        if file_path is None:
            if getattr(sys, 'frozen', False):
                # Running as compiled exe - save to temp directory
                import tempfile
                temp_dir = tempfile.gettempdir()
                file_path = os.path.join(temp_dir, 'spam_model.pkl')
            else:
                # Running as script
                base_path = os.path.dirname(os.path.abspath(__file__))
                file_path = os.path.join(base_path, 'data', 'spam_model.pkl')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        model_data = {
            'classifier': self.classifier,
            'feature_extractor': self.feature_extractor,
            'preprocessor': self.preprocessor,
            'is_model_trained': self.is_model_trained
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def classify_email(self):
        """
        Classify the email text as spam or ham using the trained model.
        
        This method:
        1. Validates that the model is trained and ready
        2. Checks that email text has been provided
        3. Uses the adaptive learning system for prediction
        4. Updates the GUI with results and confidence scores
        5. Applies color coding (red for spam, green for ham)
        6. Stores prediction data for potential user feedback
        
        Raises:
            ValueError: If model is not trained
            UserWarning: If no email text is provided
        """
        if not self.is_model_trained:
            messagebox.showerror("Error", "Model is not trained.")
            return
        
        email_text = self.email_text.get('1.0', tk.END).strip()
        
        if not email_text or email_text == "Paste your email content here for analysis...":
            messagebox.showwarning("Warning", "Please enter email text to analyze.")
            return
        
        try:
            self.status_var.set("🔍 Analyzing email...")
            self.root.update()
            
            # Use adaptive system for prediction
            result = self.adaptive_system.predict_with_feedback(email_text)
            
            # Update UI with modern styling
            prediction = "🚨 SPAM" if result['prediction'] == 1 else "✅ HAM"
            confidence = result['confidence']
            
            self.prediction_label.config(text=f"Status: {prediction}")
            self.confidence_label.config(text=f"Confidence: {confidence:.1%}")
            
            # Color coding with better colors
            if result['prediction'] == 1:
                self.prediction_label.config(foreground='#e74c3c')  # Red
                self.confidence_label.config(foreground='#e74c3c')
            else:
                self.prediction_label.config(foreground='#27ae60')  # Green
                self.confidence_label.config(foreground='#27ae60')
            
            # Store for feedback
            self.current_prediction = result
            
            self.status_var.set("✅ Analysis completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed: {str(e)}")
            self.status_var.set("Classification failed")
    
    def submit_feedback(self, is_correct):
        """
        Submit user feedback on the current prediction.
        
        Args:
            is_correct (bool): True if the prediction was correct, False otherwise
            
        This method:
        1. Validates that there is a current prediction to provide feedback for
        2. Calculates the user's correction based on the feedback
        3. Submits feedback to the adaptive learning system
        4. Automatically saves the model after feedback submission
        5. Shows success/error messages to the user
        
        The feedback is used to improve the model through adaptive learning.
        """
        if not self.current_prediction:
            messagebox.showwarning("Warning", "No prediction to provide feedback for.")
            return
        
        try:
            user_correction = self.current_prediction['prediction'] if is_correct else (1 - self.current_prediction['prediction'])
            
            result = self.adaptive_system.submit_feedback(
                self.current_prediction['email_text'],
                self.current_prediction['prediction'],
                user_correction,
                self.current_prediction['confidence']
            )
            
            if result['feedback_submitted']:
                messagebox.showinfo("Success", "Feedback submitted successfully!")
                # Auto-save model after feedback
                self.save_model()
            else:
                messagebox.showerror("Error", "Failed to submit feedback.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to submit feedback: {str(e)}")
    
    def retrain_model(self):
        """
        Retrain the model using the latest data and user feedback.
        
        This method:
        1. Updates the status to show retraining is in progress
        2. Creates a new model using current data and feedback
        3. Saves the retrained model to disk
        4. Updates the GUI to reflect the new model status
        5. Shows success/error messages to the user
        
        The retraining process improves model accuracy based on user feedback
        and ensures the system continues to learn and adapt.
        """
        try:
            self.status_var.set("Retraining model...")
            self.root.update()
            
            # Create new model with SpamAssassin-style data
            self.create_model()
            
            # Save the model
            self.save_model()
            
            messagebox.showinfo("Success", "Model retrained successfully!")
            self.status_var.set("Model retrained successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Retraining failed: {str(e)}")
            self.status_var.set("Error during retraining")

def main():
    """
    Main function to run the Spam Detection System application.
    
    This function:
    1. Creates the data directory if it doesn't exist
    2. Initializes the main Tkinter window
    3. Creates the SpamDetectionGUI application instance
    4. Starts the GUI event loop
    
    The application will run until the user closes the window.
    """
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Create main window
    root = tk.Tk()
    app = SpamDetectionGUI(root)
    
    # Start the application
    root.mainloop()

if __name__ == "__main__":
    main()
