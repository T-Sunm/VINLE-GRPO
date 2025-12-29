# src/rewards/base_reward.py
import torch
from torchmetrics.text import BERTScore
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class BaseRewardScorer:
    """
    Base class chứa shared BERTScore model để tái sử dụng cho nhiều reward functions.
    """
    _shared_bertscore = None
    _device = None
    
    @classmethod
    def get_bertscore_metric(cls):
        """Lazy initialization của shared BERTScore model"""
        if cls._shared_bertscore is None:
            cls._device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Initializing shared BERTScore (PhoBERT) on device: {cls._device}")
            cls._shared_bertscore = BERTScore(
                model_name_or_path="/mnt/dataset1/pretrained_fm/vinai/phobert-base",
                num_layers=12,
                rescale_with_baseline=False, 
                device=cls._device
            )
            print("Shared BERTScore initialized.")
        return cls._shared_bertscore
    
    @classmethod
    def calculate_bertscore_single(cls, prediction: str, ground_truth: str) -> float:
        """
        Tính BERTScore cho một cặp prediction-ground_truth.
        
        Args:
            prediction: Câu dự đoán
            ground_truth: Câu tham chiếu
            
        Returns:
            BERTScore F1 (float)
        """
        pred = str(prediction).strip()
        gt = str(ground_truth).strip()
        
        if not pred or not gt:
            return 0.0
        
        try:
            metric = cls.get_bertscore_metric()
            metric.reset()
            metric.update([pred], [gt])
            score_dict = metric.compute()
            return score_dict['f1'].item()
        except Exception as e:
            print(f"Error calculating BERTScore: {e}")
            return 0.0
    
    @classmethod
    def calculate_bertscore_batch(cls, ground_truths: dict, predictions: dict) -> dict:
        """
        Tính BERTScore cho batch predictions.
        
        Args:
            ground_truths: {id: [gt1, gt2, ...]} hoặc {id: gt_string}
            predictions: {id: prediction_string}
            
        Returns:
            {id: bertscore_f1}
        """
        bert_scores_dict = {}

        for id_, pred in predictions.items():
            gt = ground_truths.get(id_, [])
            
            # Xử lý GT: string hoặc list
            if isinstance(gt, str):
                gt_list = [gt.strip()] if gt.strip() else []
            else:
                gt_list = [str(g).strip() for g in gt if str(g).strip()]
            
            if not gt_list:
                bert_scores_dict[id_] = 0.0
                continue
            
            # Tính BERTScore với TẤT CẢ references và lấy max
            scores = []
            for gt_text in gt_list:
                score = cls.calculate_bertscore_single(pred, gt_text)
                scores.append(score)
            
            # Lấy score cao nhất trong các references
            bert_scores_dict[id_] = max(scores)

        return bert_scores_dict
