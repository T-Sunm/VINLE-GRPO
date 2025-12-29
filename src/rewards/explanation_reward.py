"""
Explanation reward scorer for explanation quality evaluation.

Combines BERTScore (semantic) and CLIPScore (visual-text alignment).
"""

import os
import re
import torch
from typing import List
from PIL import Image

try:
    from torchmetrics.multimodal import CLIPScore
except ImportError:
    CLIPScore = None

try:
    from .base_rewards import BaseRewardScorer
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from base_rewards import BaseRewardScorer


class ExplanationRewardScorer(BaseRewardScorer):
    """
    Explanation scorer combining BERTScore and CLIPScore.
    
    reward = alpha * BERTScore + (1 - alpha) * CLIPScore_normalized
    """
    
    def __init__(self, alpha=0.5, clip_model_name="openai/clip-vit-base-patch16"):
        super().__init__()
        self.alpha = alpha
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if CLIPScore is not None:
            self.clip_metric = CLIPScore(model_name_or_path=clip_model_name).to(self.device)
        else:
            self.clip_metric = None
            print("⚠️  CLIPScore not available")
    
    @staticmethod
    def _extract_tag_content(text: str, tag: str) -> str:
        """Extract content from XML tags."""
        match = re.search(fr'<{tag}>(.*?)</{tag}>', text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""
    
    def calculate_clip_batch(self, image_paths: dict, predictions: dict) -> dict:
        """Calculate CLIPScore for batch."""
        if self.clip_metric is None:
            return {img_id: 0.0 for img_id in image_paths.keys()}
        
        image_ids = list(predictions.keys())
        pil_images = []
        pred_texts = []
        valid_img_ids = []
        clip_scores_dict = {img_id: 0.0 for img_id in image_ids}
        
        for img_id in image_ids:
            image_path = image_paths[img_id]
            pred_text = str(predictions[img_id]).strip()
            
            if pred_text and image_path:
                try:
                    image = Image.open(image_path).convert("RGB")
                    pil_images.append(image)
                    pred_texts.append(pred_text)
                    valid_img_ids.append(img_id)
                except Exception as e:
                    print(f"⚠️  Error loading {image_path}: {e}")
        
        if not pil_images:
            return clip_scores_dict
        
        try:
            processor = self.clip_metric.processor
            model = self.clip_metric.model
            model.eval()
            
            inputs = processor(
                text=pred_texts,
                images=pil_images,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
            
            scores_tensor = logits_per_image.diag()
            
            for img_id, score in zip(valid_img_ids, scores_tensor.tolist()):
                clip_scores_dict[img_id] = score
        except Exception as e:
            print(f"Error during CLIP: {e}")
        
        return clip_scores_dict
    
    def explanation_rewards(
        self,
        ground_truths: List[List[str]],
        predictions: List[str],
        image_paths: List[str],
        prompt_ids: List[int] = None
    ) -> List[float]:
        """Calculate combined rewards for batch."""
        assert len(ground_truths) == len(predictions) == len(image_paths)
        
        if not predictions:
            return []
        
        gts_dict = {}
        preds_dict = {}
        paths_dict = {}
        
        for i, (gt_list, pred, image_path) in enumerate(zip(ground_truths, predictions, image_paths)):
            if isinstance(gt_list, list) and len(gt_list) > 0:
                gt = gt_list[0]
            elif isinstance(gt_list, str):
                gt = gt_list
            else:
                gt = ""
            
            gts_dict[i] = gt
            preds_dict[i] = pred
            paths_dict[i] = image_path
        
        # Calculate metrics
        bert_scores = self.calculate_bertscore_batch(gts_dict, preds_dict)
        clip_scores = self.calculate_clip_batch(paths_dict, preds_dict)
        
        # Combine
        final_rewards = []
        for i in range(len(predictions)):
            bert_score = bert_scores.get(i, 0.0)
            clip_score_raw = clip_scores.get(i, 0.0)
            
            # Normalize CLIP (15-35 -> 0-1)
            clip_score_normalized = max(0, (clip_score_raw - 15) / (35 - 15))
            
            reward = self.alpha * bert_score + (1.0 - self.alpha) * clip_score_normalized
            final_rewards.append(reward)
        
        return final_rewards


# Global instance
_explanation_scorer = None

def get_explanation_scorer(alpha=0.5, clip_model_name="openai/clip-vit-base-patch16"):
    """Get or create global explanation scorer."""
    global _explanation_scorer
    if _explanation_scorer is None:
        print(f"Initializing ExplanationRewardScorer (alpha={alpha})...")
        _explanation_scorer = ExplanationRewardScorer(alpha=alpha, clip_model_name=clip_model_name)
        print("✅ ExplanationRewardScorer initialized")
    return _explanation_scorer
