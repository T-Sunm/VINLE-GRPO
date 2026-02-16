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
    _shared_tokenizer = None
    _device = None
    _model_path = None
    
    # Mapping model names to HuggingFace IDs or paths
    MODEL_MAPPING = {
        'phobert': 'vinai/phobert-base',
        'bert': 'bert-base-uncased'
    }
    
    @classmethod
    def initialize_bertscore(cls, model_name_or_path="phobert"):
        """Initialize shared BERTScore and Tokenizer with bert-score library."""
        # Resolve model path
        model_path = cls.MODEL_MAPPING.get(model_name_or_path, model_name_or_path)
        
        # Check if re-initialization is needed
        if cls._shared_bertscore is None or cls._model_path != model_path:
            # For rewards, CPU is much safer to avoid CUDA asserts crashing the whole training
            # We default to CPU if specified or if we want maximum stability.
            # But here we follow the "optimized" GPU path with sanitization
            cls._device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._model_path = model_path
            
            print(f"Initializing shared BERTScore ({model_path}) on {cls._device}...")
            
            try:
                from transformers import AutoTokenizer
                cls._shared_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                
                cls._shared_bertscore = bert_score.BERTScorer(
                    model_type=model_path,
                    num_layers=12,
                    batch_size=64,
                    nthreads=4,
                    all_layers=False,
                    idf=False,
                    device=cls._device,
                    lang=None,
                    rescale_with_baseline=False
                )
                print("✅ Shared BERTScore and Tokenizer initialized.")
            except Exception as e:
                print(f"❌ Error initializing BERTScore: {e}")
                cls._shared_bertscore = None
                cls._shared_tokenizer = None
                
        return cls._shared_bertscore, cls._shared_tokenizer

    @classmethod
    def _safe_text(cls, text: str, tokenizer, max_len: int = 256) -> str:
        """
        Round-trip tokenize to ensure text is safe for BERTScore on GPU.
        Catches out-of-vocab IDs and truncates long inputs before they hit CUDA.
        """
        if not isinstance(text, str) or not text.strip():
            return "."

        # Remove control characters and non-BMP symbols
        text = ''.join(ch for ch in text if ord(ch) >= 32 and ord(ch) < 65536)
        text = " ".join(text.split()).strip()
        if not text:
            return "."

        try:
            # Round-trip tokenization
            enc = tokenizer(text, truncation=True, max_length=max_len, padding=False)
            ids = enc["input_ids"]

            vocab_size = len(tokenizer)
            if any((i < 0 or i >= vocab_size) for i in ids):
                return "."

            out = tokenizer.decode(ids, skip_special_tokens=True).strip()
            return out if out else "."
        except:
            return "."

    @classmethod
    def calculate_bertscore_batch(cls, ground_truths: dict, predictions: dict,
                                  model_name_or_path="phobert") -> dict:
        """
        Calculate BERTScore for a batch with round-trip sanitization.
        """
        ids = list(predictions.keys())
        bert_scores_dict = {id_: 0.0 for id_ in ids}
        
        scorer, tokenizer = cls.initialize_bertscore(model_name_or_path)
        if scorer is None or tokenizer is None:
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
                gt_text = str(gt[0]).strip()
            else:
                gt_text = ""
            
            if pred and gt_text:
                # Sanitize both pred and ref
                safe_pred = cls._safe_text(pred, tokenizer)
                safe_ref = cls._safe_text(gt_text, tokenizer)
                
                valid_ids.append(id_)
                preds_list.append(safe_pred)
                refs_list.append(safe_ref)
        
        if not valid_ids:
            return bert_scores_dict
        
        # Batch compute with bert_score
        try:
            with torch.no_grad():
                P, R, F1 = scorer.score(preds_list, refs_list)
                
                for i, id_ in enumerate(valid_ids):
                    bert_scores_dict[id_] = F1[i].item()
        
        except Exception as e:
            print(f"Error calculating BERTScore batch: {e}")
            
        return bert_scores_dict
