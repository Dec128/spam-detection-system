from setuptools import setup, find_packages

setup(
    name="spam-detection-system",
    version="1.0.0",
    description="A machine learning-based spam detection system",
    author="Alice Johnson, Bob Smith, Clara Lee, David Kim",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "nltk>=3.8.0",
        "spacy>=3.6.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    python_requires=">=3.8",
)
