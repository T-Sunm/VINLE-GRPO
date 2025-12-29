# src/rewards/base_rewards.py

import os
import torch
import bert_score

# Remove hardcoded CUDA_VISIBLE_DEVICES to allow config control
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class BaseRewardScorer:
    """
    Base class containing shared BERTScore model to reuse across reward functions.
    Uses 'bert_score' library directly (not torchmetrics) for better compatibility with PhoBERT.
    """
    
    _shared_bertscore = None
    _device = None
    _model_path = None
    
    # Mapping model names to HuggingFace IDs or paths
    MODEL_MAPPING = {
        'phobert': 'vinai/phobert-base',
        'bert': 'bert-base-uncased'
    }
    
    @classmethod
    def initialize_bertscore(cls, model_name_or_path="phobert"):
        """Initialize shared BERTScore model with bert-score library."""
        # Resolve model path
        model_path = cls.MODEL_MAPPING.get(model_name_or_path, model_name_or_path)
        
        # Check if re-initialization is needed
        if cls._shared_bertscore is None or cls._model_path != model_path:
            cls._device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._model_path = model_path
            
            print(f"Initializing shared BERTScore ({model_path}) on {cls._device}...")
            
            try:
                cls._shared_bertscore = bert_score.BERTScorer(
                    model_type=model_path,
                    num_layers=12,
                    batch_size=64,
                    nthreads=4,
                    all_layers=False,
                    idf=False,
                    device=cls._device,
                    lang=None,  # Auto-detect or specified by model
                    rescale_with_baseline=False
                )
                print("✅ Shared BERTScore initialized.")
            except Exception as e:
                print(f"❌ Error initializing BERTScore: {e}")
                cls._shared_bertscore = None
                
        return cls._shared_bertscore
    
    @classmethod
    def calculate_bertscore_batch(cls, ground_truths: dict, predictions: dict,
                                  model_name_or_path="phobert") -> dict:
        """
        Calculate BERTScore for a batch of predictions.
        
        Args:
            ground_truths: {id: [gt1, gt2, ...]} or {id: "gt_string"}
            predictions: {id: prediction_string}
            model_name_or_path: 'bert', 'phobert', or full HuggingFace path
            
        Returns:
            {id: bertscore_f1}
        """
        ids = list(predictions.keys())
        bert_scores_dict = {id_: 0.0 for id_ in ids}
        
        scorer = cls.initialize_bertscore(model_name_or_path)
        if scorer is None:
            return bert_scores_dict
        
        # Prepare batch data
        valid_ids = []
        preds_list = []
        refs_list = []
        
        for id_ in ids:
            pred = str(predictions[id_]).strip()
            gt = ground_truths.get(id_, [])
            
            # Format Ground Truth
            if isinstance(gt, str):
                gt_text = gt.strip()
            elif isinstance(gt, list) and len(gt) > 0:
                # bert_score supports multiple references, but here we simplify to similar structure
                # For multiple refs, we can adjust logic if needed. 
                # This implementation aligns with the reference file provided.
                gt_text = str(gt[0]).strip()
            else:
                gt_text = ""
            
            if pred and gt_text:
                valid_ids.append(id_)
                preds_list.append(pred)
                refs_list.append(gt_text)
        
        if not valid_ids:
            return bert_scores_dict
        
        # Batch compute with bert_score
        try:
            # Wrap in no_grad to prevent memory leaks/graph build-up
            with torch.no_grad():
                # score returns (P, R, F1)
                P, R, F1 = scorer.score(preds_list, refs_list)
                
                for i, id_ in enumerate(valid_ids):
                    bert_scores_dict[id_] = F1[i].item()
        
        except Exception as e:
            print(f"Error calculating BERTScore batch: {e}")
            # Fallback to 0.0 is handled by initialization of bert_scores_dict
            
        return bert_scores_dict
