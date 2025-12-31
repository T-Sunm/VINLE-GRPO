"""
NLG (Natural Language Generation) metrics for Vietnamese VQA evaluation.

This module provides functions for computing various NLG metrics:
- Traditional metrics: BLEU, METEOR, ROUGE-L, CIDEr
- BERTScore with PhoBERT
- SMILE metric for answer evaluation
"""

import numpy as np
import torch

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

from ..core.shared_models import SharedBERTScoreModel, SharedSMILEModel
from ..core.text_preprocessing import (
    clean_text,
    segment_vietnamese,
    truncate_sentence,
    preprocess_vietnamese_text,
    normalize_answer,
    sanitize_text_for_bert,
    # Standardized pipelines
    preprocess_for_nlg_metrics,
    preprocess_for_smile,
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
    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]
    
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

def compute_bertscore_max_ref(hypotheses: list[str], references: list[list[str]], 
                              device: str = "cuda", model_type: str = "bert") -> list[float]:
    """
    Compute BERTScore F1 with max over multiple references.
    
    For each hypothesis, computes BERTScore against all its references
    and returns the maximum F1 score.
    
    Uses aggressive input sanitization to prevent CUDA errors.
    """
    if not hypotheses:
        return []
    
    # Prepare all valid pairs with aggressive sanitization
    all_cands, all_refs = [], []
    sample_indices = []
    
    for idx, (hyp, refs) in enumerate(zip(hypotheses, references)):
        hyp_clean = sanitize_text_for_bert(hyp)
        valid_refs = [sanitize_text_for_bert(r) for r in refs if r and r.strip()]
        
        # Additional sanitization
        hyp_clean = ''.join(ch for ch in hyp_clean if ord(ch) < 65536)
        valid_refs = [''.join(ch for ch in ref if ord(ch) < 65536) for ref in valid_refs]
        
        if not valid_refs or hyp_clean == "." or len(hyp_clean.strip()) == 0:
            continue
        
        for ref in valid_refs:
            if ref != "." and len(ref.strip()) > 0:
                all_cands.append(hyp_clean)
                all_refs.append(ref)
                sample_indices.append(idx)
    
    max_scores = [0.0] * len(hypotheses)
    
    if not all_cands:
        return max_scores
    
    try:
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        scorer = SharedBERTScoreModel.get_scorer(model_type=model_type, device=device)
        
        # Process in batches with error handling
        batch_size = 128
        all_f1_scores = []
        
        for i in range(0, len(all_cands), batch_size):
            batch_cands = all_cands[i:i+batch_size]
            batch_refs = all_refs[i:i+batch_size]
            
            try:
                P, R, F1 = scorer.score(batch_cands, batch_refs)
                all_f1_scores.extend(F1.cpu().tolist())
            except Exception as batch_error:
                print(f"Warning: BERTScore batch {i//batch_size} failed: {batch_error}")
                # Append zeros for failed batch
                all_f1_scores.extend([0.0] * len(batch_cands))
                
                # Clear CUDA cache after error
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except:
                        pass
        
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
    
    SMILE (Sentence-level Metrics for Information-Leveraging Evaluation)
    evaluates answer quality by comparing semantic similarity and keyword overlap.
    
    Answers are normalized before evaluation to ensure variants like
    "có", "đúng", "yes", "vâng" are treated equally.
    
    Args:
        questions: List of questions
        gt_answers: List of ground truth answers
        predictions: List of predicted answers
        synthetic_answers: List of pre-generated synthetic full-sentence answers.
                           If None, ground truth answers will be used.
        model_type: "bert" or "phobert"
    
    Returns:
        Dictionary with SMILE metrics (avg, hm)
    """
    if not questions or not gt_answers or not predictions:
        return {"SMILE_avg": 0.0, "SMILE_hm": 0.0}
    
    # Normalize answers (important for yes/no questions)
    gt_answers = [normalize_answer(ans) for ans in gt_answers]
    predictions = [normalize_answer(pred) for pred in predictions]
    
    # Use ground truth answers as fallback if no synthetic answers provided
    if synthetic_answers is None:
        synthetic_answers = gt_answers
    else:
        synthetic_answers = [normalize_answer(ans) for ans in synthetic_answers]
    
    if len(synthetic_answers) != len(questions):
        print(f"Warning: synthetic_answers length ({len(synthetic_answers)}) "
              f"does not match questions length ({len(questions)}). Using GT answers.")
        synthetic_answers = gt_answers
    
    # Prepare data: segment Vietnamese text
    smile_data = []
    
    for i, (q, gt, syn_ans, pred) in enumerate(zip(questions, gt_answers, synthetic_answers, predictions)):
        # Skip if any raw field is empty or whitespace-only
        if not q or not q.strip() or not gt or not gt.strip() or not pred or not pred.strip():
            continue
        
        try:
            # Preprocess all fields using standardized pipeline
            q_seg = preprocess_for_smile(q)
            gt_seg = preprocess_for_smile(gt)
            syn_ans_seg = preprocess_for_smile(syn_ans)
            pred_seg = preprocess_for_smile(pred)
            
            # Ensure all segmented texts are non-empty
            if all(text and text.strip() for text in [q_seg, gt_seg, syn_ans_seg, pred_seg]):
                smile_data.append((q_seg, gt_seg, syn_ans_seg, pred_seg))
        except Exception as e:
            print(f"Warning: SMILE preprocessing failed for sample {i}: {e}")
    
    if not smile_data:
        return {"SMILE_avg": 0.0, "SMILE_hm": 0.0}
    
    # Clear any previous CUDA errors before SMILE computation
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except:
            pass
    
    # Compute SMILE scores with error handling
    try:
        smile = SharedSMILEModel.get_instance(model_type=model_type)
        smile_data_array = np.array(smile_data)
        results = smile.generate_scores(smile_data_array)
        
        return {
            "SMILE_avg": float(np.mean(results['avg'])) * 100,
            "SMILE_hm": float(np.mean(results['hm'])) * 100,
        }
    except (RuntimeError, torch.cuda.CudaError) as e:
        print(f"Warning: SMILE computation failed with CUDA error: {e}")
        print("Returning zero scores for SMILE metrics.")
        return {"SMILE_avg": 0.0, "SMILE_hm": 0.0}
    except Exception as e:
        print(f"Warning: SMILE computation failed: {e}")
        print("Returning zero scores for SMILE metrics.")
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
