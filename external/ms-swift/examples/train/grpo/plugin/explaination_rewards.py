import torch
import numpy as np
import re
import os
import warnings
from datetime import datetime
from PIL import Image
from torchmetrics.multimodal import CLIPScore

from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from transformers import AutoModel, AutoTokenizer
import atexit
from base_rewards import BaseRewardScorer

warnings.filterwarnings("ignore", category=UserWarning, module='transformers.modeling_utils')

import py_vncorenlp

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


_rdrsegmenter = None

def get_segmenter():
    """Lazy initialization của VnCoreNLP segmenter - chỉ load khi cần."""
    global _rdrsegmenter
    if _rdrsegmenter is None:
        vncorenlp_dir = '/home/vlai-vqa-nle/minhtq/vqa-nle/vncorenlp_models'
        if not os.path.exists(vncorenlp_dir):
            os.makedirs(vncorenlp_dir)
        _rdrsegmenter = py_vncorenlp.VnCoreNLP(
            annotators=["wseg"], 
            save_dir='/home/vlai-vqa-nle/minhtq/vqa-nle/src/inference/vncorenlp_models'
        )
    return _rdrsegmenter


def segment_text(text: str) -> str:
    """Segment Vietnamese text using VnCoreNLP."""
    if not text:
        return ""
    try:
        segmenter = get_segmenter()
        sentences = segmenter.word_segment(text)
        segmented_text = " ".join([" ".join(sentence) for sentence in sentences])
        return segmented_text
    except Exception as e:
        print(f"Error segmenting text: {e}. Text: '{text}'. Returning empty string.")
        return "" 


class ExplanationRewardScorer(BaseRewardScorer):
    """
    Reward scorer using BERTScore for semantic similarity evaluation.
    Uses PhoBERT for Vietnamese text evaluation.
    
    Note: CLIPScore functionality is commented out but preserved for future use.
    """
    def __init__(self, alpha: float = 1.0):
        """
        Initialize the reward scorer.
        
        Args:
            alpha: Weight for BERTScore (set to 1.0 to use only BERTScore)
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha must be in [0, 1]")

        self.alpha = alpha
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.cider_scorer = Cider()
        self.bertscore_metric = self.initialize_bertscore(model_name_or_path="phobert")
        
        # === CLIP INITIALIZATION (COMMENTED OUT) ===
        # clip_model_name = "/home/vlai-vqa-nle/.cache/huggingface/hub/models--openai--clip-vit-base-patch16/snapshots/57c216476eefef5ab752ec549e440a49ae4ae5f3"
        # self.clip_metric = CLIPScore(model_name_or_path=clip_model_name).to(self.device)

    def calculate_cider_batch(self, ground_truths: dict, predictions: dict) -> dict:
        """
        Calculate CIDEr scores for batch predictions.
        
        Args:
            ground_truths: {img_id: [reference_captions]}
            predictions: {img_id: prediction_string}
            
        Returns:
            {img_id: cider_score}
        """
        res = {img_id: [caption] for img_id, caption in predictions.items()}
        _, individual_scores_array = self.cider_scorer.compute_score(ground_truths, res)
        
        image_ids = list(predictions.keys())
        cider_scores_dict = {img_id: score for img_id, score in zip(image_ids, individual_scores_array)}
        return cider_scores_dict

    # === CLIP CALCULATION METHOD (COMMENTED OUT) ===
    # def calculate_clip_batch(self, image_paths: dict, predictions: dict) -> dict:
    #     """
    #     Calculate CLIPScore for batch predictions.
    #     
    #     Args:
    #         image_paths: {img_id: image_path}
    #         predictions: {img_id: prediction_string}
    #         
    #     Returns:
    #         {img_id: clip_score}
    #     """
    #     image_ids = list(predictions.keys())
    #     pil_images = []
    #     pred_captions = []
    #     valid_img_ids = []
    #     clip_scores_dict = {img_id: 0.0 for img_id in image_ids}
    #
    #     for img_id in image_ids:
    #         image_path = image_paths[img_id]
    #         pred_caption = str(predictions[img_id]).strip()
    #
    #         if pred_caption:
    #             try:
    #                 image = Image.open(image_path).convert("RGB")
    #                 pil_images.append(image)
    #                 pred_captions.append(pred_caption)
    #                 valid_img_ids.append(img_id)
    #             except FileNotFoundError:
    #                 print(f"Warning: Image not found at {image_path}. Skipping.")
    #             except Exception as e:
    #                 print(f"Warning: Error loading image {image_path}: {e}. Skipping.")
    #
    #     if not pil_images:
    #         return clip_scores_dict
    #
    #     try:
    #         processor = self.clip_metric.processor
    #         model = self.clip_metric.model
    #         model.eval()
    #
    #         inputs = processor(
    #             text=pred_captions,
    #             images=pil_images,
    #             return_tensors="pt",
    #             padding=True,
    #             truncation=True
    #         ).to(self.device)
    #
    #         with torch.no_grad():
    #             outputs = model(**inputs)
    #             logits_per_image = outputs.logits_per_image
    #
    #         scores_tensor = logits_per_image.diag()
    #
    #         for img_id, score in zip(valid_img_ids, scores_tensor.tolist()):
    #             clip_scores_dict[img_id] = score
    #
    #     except Exception as e:
    #         print(f"Error during CLIP batch computation: {e}")
    #
    #     return clip_scores_dict

    def explanation_rewards(self, ground_truths: list[str], predictions: list[str], 
                        image_paths: list[str], prompt_ids: list[int],
                        threshold: float = 0.3) -> list[float]:
        """
        Calculate BERTScore-based rewards for batch with penalty for low scores.
        
        Args:
            ground_truths: List of reference explanations
            predictions: List of generated explanations
            image_paths: List of image file paths (kept for future CLIP integration)
            prompt_ids: ID of prompt for each completion (for grouping)
            threshold: Minimum reward threshold. Scores below this receive penalty of -1.0
        
        Returns:
            List of rewards for each completion
        """
        from collections import defaultdict
        
        assert len(ground_truths) == len(predictions) == len(image_paths) == len(prompt_ids), \
            "Input lists must have the same length."
        
        if not predictions:
            return []
        
        # Prepare ground truths
        gts_dict = {}
        for i, gt in enumerate(ground_truths):
            if isinstance(gt, list):
                gt_text = gt[0] if gt else ""
            else:
                gt_text = gt
            gts_dict[i] = [segment_text(gt_text.strip())]
        
        # Prepare predictions and track empty ones
        preds_dict = {}
        empty_indices = set()
        
        for i, pred in enumerate(predictions):
            pred_segmented = segment_text(pred.strip())
            if not pred_segmented or not pred_segmented.strip():
                empty_indices.add(i)
            else:
                preds_dict[i] = pred_segmented
        
        paths_dict = {i: path for i, path in enumerate(image_paths)}
        
        # Calculate BERTScore for non-empty predictions
        bert_scores_dict = self.calculate_bertscore_batch(gts_dict, preds_dict)
        
        # Clear cache sau mỗi batch để tránh memory leak
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # === CLIP SCORE CALCULATION (COMMENTED OUT) ===
        # clip_scores_raw_dict = self.calculate_clip_batch(paths_dict, preds_dict)
        
        # === CLIP NORMALIZATION PER GROUP (COMMENTED OUT) ===
        # clip_normalized_dict = {}
        # 
        # # Group samples by prompt_id
        # groups = defaultdict(list)
        # for i in range(len(predictions)):
        #     if i not in empty_indices: 
        #         pid = prompt_ids[i]
        #         groups[pid].append(i)
        # 
        # # Normalize within each group separately
        # for pid, indices in groups.items():
        #     group_clip_scores = [clip_scores_raw_dict.get(idx, 0.0) for idx in indices]
        #     group_tensor = torch.tensor(group_clip_scores, device=self.device, dtype=torch.float32)
        #     
        #     if len(group_tensor) > 1:
        #         g_min = group_tensor.min()
        #         g_max = group_tensor.max()
        #         g_range = g_max - g_min
        #         
        #         if g_range > 1e-8:
        #             normalized = (group_tensor - g_min) / g_range
        #         else:
        #             normalized = torch.full_like(group_tensor, 0.5)
        #         
        #         print(f"   [Group {pid}] Size={len(indices)}, CLIP Min={g_min:.4f}, Max={g_max:.4f}, Range={g_range:.4f}")
        #     else:
        #         normalized = torch.full_like(group_tensor, 0.5)
        #         print(f"   [Group {pid}] Size=1, assigned CLIPNorm=0.5")
        #     
        #     for idx, i in enumerate(indices):
        #         clip_normalized_dict[i] = normalized[idx].item()
        
        # Build final rewards using only BERTScore
        final_rewards = []
        for i in range(len(predictions)):
            if i in empty_indices:
                reward = -1.0
                print(f"   [Sample {i}, Group {prompt_ids[i]}] Empty prediction -> Penalty={reward}")
            else:
                bert_score = bert_scores_dict.get(i, 0.0)
                
                # Reward is just BERTScore (alpha = 1.0)
                reward = bert_score
                
                # === COMBINED REWARD WITH CLIP (COMMENTED OUT) ===
                # clip_raw = clip_scores_raw_dict.get(i, 0.0)
                # clip_norm = clip_normalized_dict.get(i, 0.0)
                # reward = (self.alpha * bert_score) + ((1.0 - self.alpha) * clip_norm)
                
                # Apply threshold penalty
                if reward < threshold:
                    print(f"   [Sample {i}, Group {prompt_ids[i]}] BERT={bert_score:.4f} -> "
                          f"Reward={reward:.4f} < Threshold={threshold} -> PENALTY=-1.0")
                    reward = -1.0
                else:
                    print(f"   [Sample {i}, Group {prompt_ids[i]}] BERT={bert_score:.4f} -> Reward={reward:.4f}")
            
            final_rewards.append(reward)
        
        return final_rewards
