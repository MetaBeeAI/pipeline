# setup.py
from setuptools import setup, find_packages

setup(
    name="metabeeai_llm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "PyPDF2",
        "requests",
        "pandas",
        "pyyaml",
        "python-dotenv",
        "openai",
        "unidecode",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="MetaBeeAI LLM Pipeline for PDF processing and data extraction",
    keywords="pdf, llm, nlp, data extraction",
)