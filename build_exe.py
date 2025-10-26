"""
PyInstaller configuration for Spam Detection System
"""
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(__file__))

def build_exe():
    """Build the executable using PyInstaller"""
    import subprocess
    
    # PyInstaller command with optimized settings
    cmd = [
        'pyinstaller',
        '--onefile',                    # Create a single executable file
        '--windowed',                   # Hide console window (GUI app)
        '--name=SpamDetectionSystem',    # Name of the executable
        '--add-data=data;data',         # Include data directory
        '--add-data=modules;modules',   # Include modules directory
        '--add-data=spamassassin;spamassassin',  # Include SpamAssassin dataset
        '--hidden-import=nltk',         # Include NLTK
        '--hidden-import=sklearn',      # Include scikit-learn
        '--hidden-import=spacy',        # Include spaCy
        '--hidden-import=email',        # Include email module
        '--hidden-import=tkinter',      # Include tkinter
        '--hidden-import=tkinter.ttk',    # Include ttk
        '--hidden-import=tkinter.scrolledtext',  # Include scrolledtext
        '--hidden-import=tkinter.messagebox',    # Include messagebox
        '--hidden-import=tkinter.filedialog',     # Include filedialog
        '--clean',                      # Clean cache before building
        'spam_detection_system/main.py'  # Main script
    ]
    
    print("Building executable...")
    print("Command:", ' '.join(cmd))
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Build successful!")
        print("Executable created: dist/SpamDetectionSystem.exe")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

if __name__ == "__main__":
    build_exe()
