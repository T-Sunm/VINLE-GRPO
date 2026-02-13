"""
VQA Evaluation Module for VINLE-GRPO.

Provides unified evaluation for all inference output formats:
- GRPO: thinking + pred_explanation + predict
- OEA: pred_explanation + predict  
- OTA: thinking + predict
- ZEROSHOT: thinking + pred_explanation + predict
"""

from .core import (
    # Text preprocessing
    segment_vietnamese,
    clean_text,
    normalize_answer,
    normalize_explanation,
    truncate_sentence,
    ensure_list,
    preprocess_vietnamese_text,
    sanitize_text_for_bert,
    # Shared models (non-SMILE)
    SharedBERTScoreModel,
    SharedCLIPScoreModel,
    # Format detection
    detect_format,
    validate_format_consistency,
)

from .metrics import (
    # VQA accuracy
    check_accuracy,
    compute_accuracy,
    # NLG metrics
    compute_traditional_metrics,
    compute_bertscore_max_ref,
    get_nlg_scores,
    # CLIP metrics
    compute_clip_scores,
)

# Lazy imports for SMILE (requires smile package + LLM)
# Use: from src.evaluation.core import SharedSMILEModel, SharedSyntheticAnswerGenerator
# Or:  from src.evaluation.metrics import compute_smile_scores

__all__ = [
    # Core - Text preprocessing
    "segment_vietnamese",
    "clean_text",
    "normalize_answer",
    "normalize_explanation",
    "truncate_sentence",
    "ensure_list",
    "preprocess_vietnamese_text",
    "sanitize_text_for_bert",
    # Core - Shared models
    "SharedBERTScoreModel",
    "SharedCLIPScoreModel",
    # Core - Format detection
    "detect_format",
    "validate_format_consistency",
    # Metrics - VQA accuracy
    "check_accuracy",
    "compute_accuracy",
    # Metrics - NLG
    "compute_traditional_metrics",
    "compute_bertscore_max_ref",
    "get_nlg_scores",
    # Metrics - CLIP
    "compute_clip_scores",
]
