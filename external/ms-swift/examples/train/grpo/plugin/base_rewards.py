import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import bert_score


class BaseRewardScorer:
    """Base class chứa shared BERTScore model để tái sử dụng cho nhiều reward functions."""
    
    _shared_bertscore = None
    _device = None
    _model_path = None
    
    MODEL_MAPPING = {
        'phobert': 'vinai/phobert-base',
        'bert': 'bert-base-uncased'
    }
    
    @classmethod
    def initialize_bertscore(cls, model_name_or_path="bert"):
        """Khởi tạo shared BERTScore model với bert-score library."""
        model_path = cls.MODEL_MAPPING.get(model_name_or_path, model_name_or_path)
        
        if cls._shared_bertscore is None or cls._model_path != model_path:
            cls._device = "cuda:0" if torch.cuda.is_available() else "cpu"
            cls._model_path = model_path
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
        return cls._shared_bertscore
    
    @classmethod
    def calculate_bertscore_batch(cls, ground_truths: dict, predictions: dict,
                                  model_name_or_path="bert") -> dict:
        """
        Tính BERTScore cho batch predictions.
        
        Args:
            ground_truths: {id: [gt1, gt2, ...]} hoặc {id: "gt_string"}
            predictions: {id: prediction_string}
            model_name_or_path: 'bert', 'phobert', hoặc full HuggingFace path
            
        Returns:
            {id: bertscore_f1}
        """
        ids = list(predictions.keys())
        bert_scores_dict = {id_: 0.0 for id_ in ids}
        
        scorer = cls.initialize_bertscore(model_name_or_path)
        
        # Prepare batch data
        valid_ids = []
        preds_list = []
        refs_list = []
        
        for id_ in ids:
            pred = str(predictions[id_]).strip()
            gt = ground_truths.get(id_, [])
            
            if isinstance(gt, str):
                gt_text = gt.strip()
            elif isinstance(gt, list) and len(gt) > 0:
                gt_text = str(gt[0]).strip()
            else:
                gt_text = ""
            
            if pred and gt_text:
                valid_ids.append(id_)
                preds_list.append(pred)
                refs_list.append(gt_text)
        
        if not valid_ids:
            return bert_scores_dict
        
        # Batch compute với bert_score.BERTScorer - wrap trong no_grad để tránh memory leak
        with torch.no_grad():
            P, R, F1 = scorer.score(preds_list, refs_list)
            
            for i, id_ in enumerate(valid_ids):
                bert_scores_dict[id_] = F1[i].item()
            
        return bert_scores_dict