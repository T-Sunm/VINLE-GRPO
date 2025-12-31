"""
VQA accuracy metrics with fuzzy matching support.

Provides accuracy computation with support for:
- Exact matching after normalization
- Yes/no answer variants (Vietnamese + English)
- Unsorted word matching
"""

import re
from typing import Dict, List

from ..core.text_preprocessing import clean_text, normalize_answer, normalize_unsorted


# ============================================================================
# CONSTANTS
# ============================================================================

YES_SET = {"yes", "true", "correct", "có", "đúng", "vâng"}
NO_SET = {"no", "false", "incorrect", "không", "sai"}


# ============================================================================
# FUZZY MATCHING
# ============================================================================



def check_accuracy(pred_ans: str, gt_ans: str, raw_pred: str, raw_gt: str) -> bool:
    """
    Check if prediction matches ground truth with fuzzy matching.
    
    Matching strategies:
    1. Exact match after normalization
    2. Yes/no variant matching
    3. Unsorted substring matching
    
    Args:
        pred_ans: Normalized prediction
        gt_ans: Normalized ground truth
        raw_pred: Raw prediction text
        raw_gt: Raw ground truth text
        
    Returns:
        True if prediction matches ground truth
    """
    # Strategy 1: Exact match
    if pred_ans == gt_ans:
        return True
    
    if not raw_pred:
        return False
    
    # Prepare cleaned prediction
    c_pred_raw = clean_text(raw_pred).lower().strip()
    c_pred_nopunct = re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', c_pred_raw)).strip()
    c_pred_tokens = set(c_pred_raw.split())
    
    # Strategy 2: Yes/no variant matching
    if gt_ans in YES_SET and not c_pred_tokens.isdisjoint(YES_SET):
        return True
    if gt_ans in NO_SET and not c_pred_tokens.isdisjoint(NO_SET):
        return True
    
    # Strategy 3: Unsorted substring matching
    gt_unsorted = normalize_unsorted(raw_gt)
    return gt_unsorted and gt_unsorted in c_pred_nopunct


# ============================================================================
# BATCH ACCURACY COMPUTATION
# ============================================================================

def compute_accuracy(data: List[Dict]) -> Dict:
    """
    Compute VQA accuracy for a dataset.
    
    Args:
        data: List of prediction dictionaries with keys:
              - answer: ground truth answer
              - predict: predicted answer
              
    Returns:
        Dictionary with overall and by-type accuracy:
        {
            'total': int,
            'correct': int,
            'accuracy': float,
            'by_answer_type': {
                'yes/no': {...},
                'number': {...},
                'other': {...}
            }
        }
    """
    total, correct = 0, 0
    by_type = {}
    
    for item in data:
        total += 1
        
        raw_gt = item.get("answer", "")
        raw_pred = item.get("predict", "")
        gt_ans = normalize_answer(raw_gt)
        pred_ans = normalize_answer(raw_pred)
        
        ans_type = item.get("answer_type", "other")
        if ans_type not in by_type:
            by_type[ans_type] = {"total": 0, "correct": 0}
        
        by_type[ans_type]["total"] += 1
        
        if check_accuracy(pred_ans, gt_ans, raw_pred, raw_gt):
            correct += 1
            by_type[ans_type]["correct"] += 1
    
    # Compute percentages
    accuracy_pct = (correct / total * 100) if total > 0 else 0
    
    by_type_results = {}
    for ans_type, counts in by_type.items():
        by_type_results[ans_type] = {
            "total": counts["total"],
            "correct": counts["correct"],
            "accuracy": (counts["correct"] / counts["total"] * 100) if counts["total"] > 0 else 0
        }
    
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy_pct,
        "by_answer_type": by_type_results
    }


__all__ = [
    "check_accuracy",
    "compute_accuracy",
]
