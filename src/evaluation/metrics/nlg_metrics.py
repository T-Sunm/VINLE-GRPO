"""
NLG (Natural Language Generation) metrics for Vietnamese VQA evaluation.

This module provides functions for computing various NLG metrics:
- Traditional metrics: BLEU, METEOR, ROUGE-L, CIDEr
- BERTScore with PhoBERT
- SMILE metric for answer evaluation
"""

import numpy as np
import torch

import shutil





from ..core.shared_models import SharedBERTScoreModel
from ..core.text_preprocessing import (
    clean_text,
    normalize_answer,
    preprocess_for_nlg_metrics,
)


# ============================================================================
# TRADITIONAL NLG METRICS
# ============================================================================

def compute_traditional_metrics(gts: dict, res: dict) -> dict[str, float]:
    """
    Compute BLEU, METEOR, ROUGE, CIDEr scores.
    
    Args:
        gts: Ground truth dict {id: [ref1, ref2, ...]}
        res: Predictions dict {id: [pred]}
        
    Returns:
        Dictionary with metric scores (scaled to 0-100)
    """
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider

    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]

    # Meteor requires Java - make it optional
    java_available = shutil.which('java') is not None
    if java_available:
        try:
            from pycocoevalcap.meteor.meteor import Meteor
            scorers.insert(1, (Meteor(), "METEOR"))
        except ImportError:
            print("Warning: pycocoevalcap.meteor not found, METEOR metric will be skipped.")
    else:
        print("Warning: Java not found, METEOR metric will be skipped.")
    
    scores = {}
    for scorer, method in scorers:
        try:
            score, _ = scorer.compute_score(gts, res)
            if isinstance(method, list):
                for m, s in zip(method, score):
                    scores[m] = float(s) * 100
            else:
                scores[method] = float(score) * 100
        except Exception:
            if isinstance(method, list):
                scores.update({m: 0.0 for m in method})
            else:
                scores[method] = 0.0
    
    return scores


# ============================================================================
# BERTSCORE
# ============================================================================

def safe_text_for_bertscore(text: str, tokenizer, max_len: int = 256) -> str:
    """
    Round-trip tokenize to ensure text is safe for BERTScore on GPU.
    Catches out-of-vocab IDs and truncates long inputs before they hit CUDA.
    """
    if not isinstance(text, str) or not text.strip():
        return "."

    text = ''.join(ch for ch in text if ord(ch) >= 32 and ord(ch) < 65536)
    text = " ".join(text.split()).strip()
    if not text:
        return "."

    enc = tokenizer(text, truncation=True, max_length=max_len, padding=False)
    ids = enc["input_ids"]

    vocab_size = len(tokenizer)
    if any((i < 0 or i >= vocab_size) for i in ids):
        return "."

    out = tokenizer.decode(ids, skip_special_tokens=True).strip()
    return out if out else "."


def compute_bertscore_max_ref(hypotheses: list[str], references: list[list[str]], 
                              device: str = "cuda", model_type: str = "bert") -> list[float]:
    """
    Compute BERTScore F1 with max over multiple references.
    
    Uses round-trip tokenization to prevent CUDA errors from out-of-vocab tokens.
    """
    if not hypotheses:
        return []
    
    # Load tokenizer for round-trip sanitization
    tokenizer = SharedBERTScoreModel.get_tokenizer(model_type=model_type)
    
    # Prepare all valid pairs with round-trip tokenize sanitization
    all_cands, all_refs = [], []
    sample_indices = []
    
    for idx, (hyp, refs) in enumerate(zip(hypotheses, references)):
        hyp_clean = safe_text_for_bertscore(hyp, tokenizer)
        valid_refs = [safe_text_for_bertscore(r, tokenizer) for r in refs if r and r.strip()]
        
        if not valid_refs or hyp_clean == ".":
            continue
        
        for ref in valid_refs:
            if ref != ".":
                all_cands.append(hyp_clean)
                all_refs.append(ref)
                sample_indices.append(idx)
    
    max_scores = [0.0] * len(hypotheses)
    
    if not all_cands:
        return max_scores
    
    try:
        scorer = SharedBERTScoreModel.get_scorer(model_type=model_type, device=device)
        
        # Process in batches
        batch_size = 128
        all_f1_scores = []
        
        for i in range(0, len(all_cands), batch_size):
            batch_cands = all_cands[i:i+batch_size]
            batch_refs = all_refs[i:i+batch_size]
            
            try:
                P, R, F1 = scorer.score(batch_cands, batch_refs)
                all_f1_scores.extend(F1.tolist())
            except Exception as batch_error:
                print(f"Warning: BERTScore batch {i//batch_size} failed: {batch_error}")
                all_f1_scores.extend([0.0] * len(batch_cands))
        
        # Assign scores
        for i, f1 in zip(sample_indices, all_f1_scores):
            max_scores[i] = max(max_scores[i], f1 * 100)
    except Exception as e:
        print(f"Warning: BERTScore computation failed: {e}")
    
    return max_scores


# ============================================================================
# COMBINED NLG SCORES
# ============================================================================

def get_nlg_scores(references: list[list[str]], hypotheses: list[str], 
                   device: str = "cuda", max_len: int = 150, model_type: str = "bert") -> dict[str, float]:
    """
    Compute all NLG metrics for Vietnamese text.
    
    Includes preprocessing with Vietnamese word segmentation.
    
    Args:
        references: List of reference lists
        hypotheses: List of predictions
        device: Device for BERTScore computation
        max_len: Maximum words per text (for truncation)
        model_type: "bert" or "phobert" for BERTScore
        
    Returns:
        Dictionary with all metric scores
    """
    # Preprocess all texts using standardized pipeline
    hypotheses = [preprocess_for_nlg_metrics(h, max_len) for h in hypotheses]
    references = [[preprocess_for_nlg_metrics(r, max_len) for r in refs] for refs in references]
    
    # Prepare data for traditional metrics
    gts = {i: [clean_text(r) for r in refs] for i, refs in enumerate(references)}
    res = {i: [clean_text(hyp)] for i, hyp in enumerate(hypotheses)}
    
    # Compute traditional metrics
    scores = compute_traditional_metrics(gts, res)
    
    # Compute BERTScore
    max_f1_scores = compute_bertscore_max_ref(hypotheses, references, device, model_type=model_type)
    scores["BERTScore_F1"] = (sum(max_f1_scores) / len(max_f1_scores)) if max_f1_scores else 0.0
    
    return scores


# ============================================================================
# SMILE METRIC
# ============================================================================



def compute_smile_scores(questions: list[str], gt_answers: list[str], 
                         predictions: list[str], 
                         synthetic_answers: list[str] = None,
                         model_type: str = "bert") -> dict[str, float]:
    """
    Compute SMILE scores for answer evaluation.
    Uses round-trip tokenization to ensure all inputs are safe for the model.
    """
    if not questions or not gt_answers or not predictions:
        return {"SMILE_avg": 0.0, "SMILE_hm": 0.0}
    
    # Get tokenizer for round-trip sanitization
    tokenizer = SharedBERTScoreModel.get_tokenizer(model_type=model_type)
    
    # Normalize answers (important for yes/no questions)
    gt_answers = [normalize_answer(ans) for ans in gt_answers]
    predictions = [normalize_answer(pred) for pred in predictions]
    
    if synthetic_answers is None:
        synthetic_answers = gt_answers
    else:
        synthetic_answers = [normalize_answer(ans) for ans in synthetic_answers]
    
    if len(synthetic_answers) != len(questions):
        print(f"Warning: synthetic_answers length ({len(synthetic_answers)}) "
              f"does not match questions length ({len(questions)}). Using GT answers.")
        synthetic_answers = gt_answers
    
    # Prepare data with round-trip tokenize sanitization
    smile_data = []
    
    for q, gt, syn_ans, pred in zip(questions, gt_answers, synthetic_answers, predictions):
        q_safe = safe_text_for_bertscore(q, tokenizer)
        gt_safe = safe_text_for_bertscore(gt, tokenizer)
        syn_safe = safe_text_for_bertscore(syn_ans, tokenizer)
        pred_safe = safe_text_for_bertscore(pred, tokenizer)
        
        if all(t != "." for t in [q_safe, gt_safe, syn_safe, pred_safe]):
            smile_data.append((q_safe, gt_safe, syn_safe, pred_safe))
    
    if not smile_data:
        return {"SMILE_avg": 0.0, "SMILE_hm": 0.0}
    
    try:
        from ..core.shared_models import SharedSMILEModel
        smile = SharedSMILEModel.get_instance(model_type=model_type)
        smile_data_array = np.array(smile_data)
        results = smile.generate_scores(smile_data_array)
        
        return {
            "SMILE_avg": float(np.mean(results['avg'])) * 100,
            "SMILE_hm": float(np.mean(results['hm'])) * 100,
        }
    except Exception as e:
        print(f"Warning: SMILE computation failed: {e}")
        return {"SMILE_avg": 0.0, "SMILE_hm": 0.0}


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "compute_traditional_metrics",
    "compute_bertscore_max_ref",
    "get_nlg_scores",
    "compute_smile_scores",
]
