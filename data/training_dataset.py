"""
Training Dataset for Spam Detection System
Contains sample emails for training the spam detection model
Supports both sample data and SpamAssassin Public Corpus
"""

import os
import email
import glob
import re
from typing import List, Tuple, Optional

# Note: Sample data has been removed - system now uses SpamAssassin Public Corpus

def get_training_data():
    """Get training data and labels - now returns SpamAssassin sample data"""
    # Create simple fallback data when SpamAssassin is not available
    sample_emails = [
        "Subject: Test spam email\n\nThis is a test spam email for fallback purposes.",
        "Subject: Test ham email\n\nThis is a test ham email for fallback purposes."
    ]
    sample_labels = [1, 0]  # 1 spam, 1 ham
    return sample_emails, sample_labels

def get_dataset_stats():
    """Get dataset statistics"""
    emails, labels = get_training_data()
    return {
        'total_emails': len(emails),
        'spam_count': labels.count(1),
        'ham_count': labels.count(0),
        'spam_ratio': labels.count(1) / len(labels) if labels else 0,
        'ham_ratio': labels.count(0) / len(labels) if labels else 0
    }

def print_dataset_info():
    """Print dataset information"""
    stats = get_dataset_stats()
    print("Training Dataset Information:")
    print("=" * 40)
    print(f"Total emails: {stats['total_emails']}")
    print(f"Spam emails: {stats['spam_count']}")
    print(f"Ham emails: {stats['ham_count']}")
    print(f"Spam ratio: {stats['spam_ratio']:.2%}")
    print(f"Ham ratio: {stats['ham_ratio']:.2%}")

def load_spamassassin_data(dataset_path: str, max_emails_per_category: int = 100) -> Tuple[List[str], List[int]]:
    """
    Load SpamAssassin Public Corpus dataset
    
    Args:
        dataset_path: Path to SpamAssassin dataset directory
        max_emails_per_category: Maximum emails per category to load
        
    Returns:
        Tuple[List[str], List[int]]: (email contents, labels)
    """
    emails = []
    labels = []
    
    # Define dataset directories and corresponding labels
    categories = [
        ('easy_ham', 0),    # Normal emails
        ('hard_ham', 0),    # Hard to classify normal emails
        ('spam', 1)         # Spam emails
    ]
    
    for category, label in categories:
        category_path = os.path.join(dataset_path, category)
        
        if not os.path.exists(category_path):
            print(f"Warning: Directory {category_path} does not exist, skipping...")
            continue
            
        # Get all email files in this category
        email_files = glob.glob(os.path.join(category_path, '*'))
        email_files = [f for f in email_files if os.path.isfile(f)]
        
        # Limit the number of emails per category
        if max_emails_per_category and len(email_files) > max_emails_per_category:
            email_files = email_files[:max_emails_per_category]
        
        print(f"Loading {category}: {len(email_files)} emails")
        
        # Load email content
        for email_file in email_files:
            try:
                email_content = _load_email_file(email_file)
                if email_content:
                    emails.append(email_content)
                    labels.append(label)
            except Exception as e:
                print(f"Failed to load email {email_file}: {e}")
                continue
    
    print(f"Total loaded: {len(emails)} emails")
    print(f"Spam emails: {labels.count(1)}")
    print(f"Ham emails: {labels.count(0)}")
    
    return emails, labels

def _load_email_file(file_path: str) -> Optional[str]:
    """Load a single email file and extract content"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Parse email
        msg = email.message_from_string(content)
        
        # Extract email content
        email_content = _extract_email_content(msg)
        
        return email_content
        
    except Exception as e:
        print(f"Error parsing email {file_path}: {e}")
        return None

def _extract_email_content(msg) -> str:
    """Extract content from email message"""
    content = ""
    
    # Extract subject
    subject = msg.get('Subject', '')
    if subject:
        content += f"Subject: {subject}\n"
    
    # Extract sender
    sender = msg.get('From', '')
    if sender:
        content += f"From: {sender}\n"
    
    # Extract email body
    body = _get_email_body(msg)
    if body:
        content += f"\n{body}"
    
    return content

def _get_email_body(msg) -> str:
    """Get email body content"""
    if msg.is_multipart():
        # Multipart email
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == 'text/plain':
                payload = part.get_payload(decode=True)
                if payload:
                    try:
                        return payload.decode('utf-8', errors='ignore')
                    except:
                        return str(payload, errors='ignore')
            elif content_type == 'text/html':
                # If HTML, extract text content
                payload = part.get_payload(decode=True)
                if payload:
                    try:
                        html_content = payload.decode('utf-8', errors='ignore')
                        # Simple HTML tag removal
                        text_content = re.sub(r'<[^>]+>', '', html_content)
                        return text_content
                    except:
                        return str(payload, errors='ignore')
    else:
        # Single part email
        content_type = msg.get_content_type()
        if content_type in ['text/plain', 'text/html']:
            payload = msg.get_payload(decode=True)
            if payload:
                try:
                    content = payload.decode('utf-8', errors='ignore')
                    if content_type == 'text/html':
                        # Remove HTML tags
                        content = re.sub(r'<[^>]+>', '', content)
                    return content
                except:
                    return str(payload, errors='ignore')
    
    return ""

def get_spamassassin_sample_data() -> Tuple[List[str], List[int]]:
    """
    Get SpamAssassin-style sample data (when real dataset is not available)
    
    Returns:
        Tuple[List[str], List[int]]: (email contents, labels)
    """
    try:
        from .spamassassin_loader import create_spamassassin_sample_data
        return create_spamassassin_sample_data()
    except ImportError:
        print("SpamAssassin loader not available, using basic sample data")
        return get_training_data()
    except Exception as e:
        print(f"Error loading SpamAssassin sample data: {e}")
        return get_training_data()

def get_enhanced_training_data(use_spamassassin: bool = False, dataset_path: str = None) -> Tuple[List[str], List[int]]:
    """
    Get enhanced training data with SpamAssassin support
    
    Args:
        use_spamassassin: Whether to use SpamAssassin dataset
        dataset_path: Path to SpamAssassin dataset (if use_spamassassin is True)
        
    Returns:
        Tuple[List[str], List[int]]: (email contents, labels)
    """
    if use_spamassassin and dataset_path:
        return load_spamassassin_data(dataset_path)
    elif use_spamassassin:
        return get_spamassassin_sample_data()
    else:
        return get_training_data()

if __name__ == "__main__":
    print_dataset_info()
    
    print("\n" + "=" * 50)
    print("SpamAssassin Dataset Support:")
    print("=" * 50)
    
    # Test SpamAssassin sample data
    print("Testing SpamAssassin sample data:")
    try:
        spamassassin_emails, spamassassin_labels = get_spamassassin_sample_data()
        print(f"Loaded {len(spamassassin_emails)} emails from SpamAssassin sample")
        print(f"Spam: {spamassassin_labels.count(1)}, Ham: {spamassassin_labels.count(0)}")
    except Exception as e:
        print(f"Error loading SpamAssassin sample data: {e}")
    
    print("\nTo use real SpamAssassin dataset:")
    print("1. Download SpamAssassin Public Corpus")
    print("2. Extract to a directory")
    print("3. Use: load_spamassassin_data('/path/to/spamassassin')")
