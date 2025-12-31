"""
Text preprocessing utilities for Vietnamese VQA evaluation.

This module provides functions for cleaning, normalizing, and preprocessing
Vietnamese text for NLG metrics computation.
"""

import re
import unicodedata

from underthesea import text_normalize, word_tokenize


# ============================================================================
# VIETNAMESE TEXT SEGMENTATION
# ============================================================================

def segment_vietnamese(text: str) -> str:
    """
    Segment Vietnamese text using underthesea word tokenizer.
    
    Args:
        text: Input Vietnamese text
        
    Returns:
        Segmented text with compound words joined by underscores
        
    Example:
        >>> segment_vietnamese("Đây là một ví dụ")
        "Đây là một ví_dụ"
    """
    if not text or not text.strip():
        return ""
    return word_tokenize(text, format="text")


# ============================================================================
# TEXT CLEANING
# ============================================================================

def clean_text(text: str) -> str:
    """
    Remove line breaks, control characters, and normalize whitespace.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text with normalized whitespace
    """
    if not text:
        return ""
    
    text = text.replace("|||", " ").replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cc")
    
    return re.sub(r"\s+", " ", text).strip()


# ============================================================================
# ANSWER NORMALIZATION
# ============================================================================

def normalize_answer(text: str) -> str:
    """
    Normalize answer for exact matching.
    
    Handles:
    - Text cleaning and lowercasing
    - Boolean answer normalization (yes/no variants)
    - Punctuation removal
    - Word sorting for order-invariant comparison
    
    Args:
        text: Raw answer text
        
    Returns:
        Normalized answer string
    """
    if not text:
        return ""
    
    text = clean_text(text).lower().strip().rstrip(".").replace('"', "").strip()
    
    # Normalize boolean answers
    if text in ["có", "đúng", "vâng", "yes", "true", "correct"]:
        return "có"
    if text in ["không", "sai", "no", "false", "incorrect"]:
        return "không"
    
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return " ".join(sorted(text.split()))


def normalize_explanation(text: str) -> str:
    """
    Normalize explanation text.
    
    Args:
        text: Raw explanation text
        
    Returns:
        Normalized explanation string
    """
    text = clean_text(text).strip().rstrip(".").strip()
    text = text.lower()
    
    return text


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def truncate_sentence(text: str, max_words: int) -> str:
    """
    Truncate sentence to maximum number of words.
    
    Args:
        text: Input text
        max_words: Maximum number of words to keep
        
    Returns:
        Truncated text
    """
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def ensure_list(value) -> list[str]:
    """
    Convert value to list of strings.
    
    Args:
        value: Input value (None, str, list, or other)
        
    Returns:
        List of strings
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)]


def preprocess_vietnamese_text(text: str) -> str:
    """
    Preprocess Vietnamese text using underthesea.
    
    Pipeline:
    1. Text normalization (fix encoding, typos)
    2. Word tokenization (segmentation)
    
    Args:
        text: Raw Vietnamese text
        
    Returns:
        Preprocessed and tokenized text
    """
    if not text or not text.strip():
        return ""
    
    normalized_text = text_normalize(text)
    tokenized_text = word_tokenize(normalized_text, format="text")
    
    return tokenized_text


def sanitize_text_for_bert(text: str) -> str:
    """
    Sanitize text for BERT-based models to prevent CUDA errors.
    
    Removes null bytes, control characters, surrogate pairs, and handles empty strings.
    Uses "." as fallback for empty strings (valid token in all vocabularies).
    
    Args:
        text: Input text
        
    Returns:
        Sanitized text safe for BERT tokenization
    """
    if not text or not text.strip():
        return "."
    
    text = text.replace('\x00', '')
    text = ''.join(ch for ch in text if ord(ch) < 65536)
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Cc' or ch in '\n\r\t ')
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    text = ' '.join(text.split())
    
    return text if text else "."


# ============================================================================
# STANDARDIZED PREPROCESSING PIPELINES
# ============================================================================

def normalize_unsorted(text: str) -> str:
    """
    Normalize text without word sorting for fuzzy matching.
    
    Used for: VQA accuracy fuzzy matching
    Pipeline: clean -> lowercase -> remove punctuation -> normalize whitespace
    
    Args:
        text: Input text
        
    Returns:
        Normalized text with punctuation removed
    """
    if not text:
        return ""
    text = clean_text(text).lower().strip().rstrip(".").replace('"', "").strip()
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def preprocess_for_nlg_metrics(text: str, max_len: int = 150) -> str:
    """
    Preprocess text for NLG metrics (BLEU, METEOR, ROUGE, CIDEr, BERTScore).
    
    Pipeline: truncate -> Vietnamese segmentation -> sanitize for BERT
    
    Args:
        text: Input text
        max_len: Maximum words to keep
        
    Returns:
        Preprocessed text ready for NLG metrics
    """
    text = truncate_sentence(text, max_len)
    text = preprocess_vietnamese_text(text)
    text = sanitize_text_for_bert(text)
    return text


def preprocess_for_smile(text: str) -> str:
    """
    Preprocess text for SMILE metric.
    
    Pipeline: validate -> remove null bytes -> remove non-BMP chars -> clean -> segment
    Returns empty string for invalid input (caller should skip).
    
    Args:
        text: Input text
        
    Returns:
        Preprocessed text or empty string if invalid
    """
    if not text or not text.strip():
        return ""
    
    # Remove null bytes
    text = text.replace('\x00', '').strip()
    if not text:
        return ""
    
    # Remove chars outside BMP (emoji, special symbols) - PhoBERT can't handle them
    text = ''.join(ch for ch in text if ord(ch) < 65536)
    if not text.strip():
        return ""
    
    text = clean_text(text)
    if not text:
        return ""
    
    text = segment_vietnamese(text)
    return text if text and text.strip() else ""


__all__ = [
    "segment_vietnamese",
    "clean_text",
    "normalize_answer",
    "normalize_explanation",
    "truncate_sentence",
    "ensure_list",
    "preprocess_vietnamese_text",
    "sanitize_text_for_bert",
    # Standardized pipelines
    "normalize_unsorted",
    "preprocess_for_nlg_metrics",
    "preprocess_for_smile",
]
