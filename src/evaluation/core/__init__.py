"""
Core utilities for VQA evaluation.
"""

from .text_preprocessing import (
    segment_vietnamese,
    clean_text,
    normalize_answer,
    normalize_explanation,
    truncate_sentence,
    ensure_list,
    preprocess_vietnamese_text,
    sanitize_text_for_bert,
)

from .shared_models import (
    SharedBERTScoreModel,
    SharedSMILEModel,
    SharedSyntheticAnswerGenerator,
)

from .format_detector import (
    detect_format,
    validate_format_consistency,
)

__all__ = [
    # Text preprocessing
    "segment_vietnamese",
    "clean_text",
    "normalize_answer",
    "normalize_explanation",
    "truncate_sentence",
    "ensure_list",
    "preprocess_vietnamese_text",
    "sanitize_text_for_bert",
    # Standardized preprocessing pipelines
    "normalize_unsorted",
    "preprocess_for_nlg_metrics",
    "preprocess_for_smile",
    # Shared models
    "SharedBERTScoreModel",
    "SharedSMILEModel",
    "SharedSyntheticAnswerGenerator",
    # Format detection
    "detect_format",
    "validate_format_consistency",
]
