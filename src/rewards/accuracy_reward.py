"""
Accuracy reward scorer for answer evaluation.

Uses hybrid ROUGE-L + BERTScore (PhoBERT).
"""

import re
from typing import Dict, List
from pycocoevalcap.rouge.rouge import Rouge

try:
    from .base_rewards import BaseRewardScorer
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from base_rewards import BaseRewardScorer


class AccuracyRewardScorer(BaseRewardScorer):
    """
    Accuracy scorer using hybrid ROUGE-L + BERTScore.
    
    reward = alpha * BERTScore + (1 - alpha) * ROUGE-L
    """
    
    def __init__(self, alpha=0.5):
        super().__init__()
        self.rouge_scorer = Rouge()
        self.alpha = alpha
    
    def calculate_rouge_batch(self, ground_truths: Dict, predictions: Dict) -> Dict:
        """Calculate ROUGE-L for batch."""
        if not predictions:
            return {}
        
        gts = {}
        res = {}
        
        for id_, pred in predictions.items():
            gt = ground_truths.get(id_, "")
            if isinstance(gt, str):
                gt_text = gt.strip()
            else:
                gt_text = gt[0].strip() if gt else ""
            
            gts[id_] = [gt_text.lower()] if gt_text else [""]
            res[id_] = [pred.lower().strip()]
        
        try:
            avg_score, individual_scores = self.rouge_scorer.compute_score(gts, res)
            return {id_: score for id_, score in zip(gts.keys(), individual_scores)}
        except Exception as e:
            print(f"Error calculating ROUGE-L: {e}")
            return {id_: 0.0 for id_ in predictions.keys()}
    
    def accuracy_rewards_batch(self, completions: List[str], solutions: List[str]) -> List[float]:
        """Calculate hybrid rewards for batch."""
        if not completions:
            return []
        
        gts_dict = {}
        preds_dict = {}
        
        for i, (completion, solution) in enumerate(zip(completions, solutions)):
            # Extract from CONCLUSION tags
            sol_match = re.search(r"<CONCLUSION>(.*?)</CONCLUSION>", solution, flags=re.DOTALL | re.IGNORECASE)
            ground_truth = sol_match.group(1).strip() if sol_match else solution.strip()
            
            content_match = re.search(r"<CONCLUSION>(.*?)</CONCLUSION>", completion, flags=re.DOTALL | re.IGNORECASE)
            student_answer = content_match.group(1).strip() if content_match else ""
            
            gts_dict[i] = ground_truth
            preds_dict[i] = student_answer
        
        # Calculate both metrics
        rouge_scores = self.calculate_rouge_batch(gts_dict, preds_dict)
        bert_scores = self.calculate_bertscore_batch(gts_dict, preds_dict)
        
        # Combine
        rewards = []
        for i in range(len(completions)):
            rouge_score = rouge_scores.get(i, 0.0)
            bert_score = bert_scores.get(i, 0.0)
            reward = self.alpha * bert_score + (1.0 - self.alpha) * rouge_score
            rewards.append(reward)
        
        return rewards


# Global instance
_accuracy_scorer = None

def get_accuracy_scorer(alpha=0.5):
    """Get or create global accuracy scorer."""
    global _accuracy_scorer
    if _accuracy_scorer is None:
        print(f"Initializing AccuracyRewardScorer (alpha={alpha})...")
        _accuracy_scorer = AccuracyRewardScorer(alpha=alpha)
        print("✅ AccuracyRewardScorer initialized")
    return _accuracy_scorer
