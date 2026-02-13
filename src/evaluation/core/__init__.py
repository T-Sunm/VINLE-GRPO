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
    SharedCLIPScoreModel,
)

from .format_detector import (
    detect_format,
    validate_format_consistency,
)

# Lazy imports for SMILE (use directly when needed):
#   from .core.shared_models import SharedSMILEModel, SharedSyntheticAnswerGenerator

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
    # Shared models
    "SharedBERTScoreModel",
    "SharedCLIPScoreModel",
    # Format detection
    "detect_format",
    "validate_format_consistency",
]
