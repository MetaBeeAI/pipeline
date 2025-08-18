"""
LLM Benchmarking Module

This module provides tools for merging LLM-generated answers with human reviewer answers
and analyzing reviewer ratings for benchmarking purposes.

Main Components:
- merge_answers.py: Merges LLM and reviewer answers from multiple sources
- reviewer_rating.py: Analyzes reviewer ratings and generates statistical plots
"""

from .merge_answers import merge_answers_from_reviewers
from .reviewer_rating import (
    calculate_question_stats,
    calculate_reviewer_agreement,
    calculate_reviewer_individual_stats,
    plot_question_ratings,
    plot_reviewer_agreement,
    plot_individual_reviewer_ratings
)
from .process_benchmarking import BenchmarkingDataProcessor

__version__ = "1.0.0"
__author__ = "MetaBeeAI Pipeline"

__all__ = [
    "merge_answers_from_reviewers",
    "calculate_question_stats",
    "calculate_reviewer_agreement", 
    "calculate_reviewer_individual_stats",
    "plot_question_ratings",
    "plot_reviewer_agreement",
    "plot_individual_reviewer_ratings",
    "BenchmarkingDataProcessor"
]
