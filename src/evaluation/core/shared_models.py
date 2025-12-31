"""
Shared singleton models for evaluation metrics.

This module provides singleton pattern implementations for expensive-to-initialize models:
- SharedBERTScoreModel: PhoBERT-based BERTScore for Vietnamese
- SharedSMILEModel: SMILE metric for answer evaluation  
- SharedSyntheticAnswerGenerator: LLM-based synthetic answer generation
"""

import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# Get repo root directory dynamically
REPO_ROOT = Path(__file__).resolve().parents[3]
SMILE_PATH = str(REPO_ROOT / 'external' / 'smile-metric')

# Add SMILE metric path
if SMILE_PATH not in sys.path:
    sys.path.append(SMILE_PATH)

from smile.smile import SMILE



# ============================================================================
# SHARED BERTSCORE MODEL
# ============================================================================

class SharedBERTScoreModel:
    
    _scorers = {}
    
    MODEL_MAPPING = {
        'phobert': 'vinai/phobert-base',
        'bert': 'bert-base-uncased'
    }
    
    @classmethod
    def _sanitize_for_bertscore(cls, text: str) -> str:
        """Remove problematic characters."""
        if not text or not isinstance(text, str):
            return "."
        
        text = ''.join(ch for ch in text if ord(ch) >= 32 and ord(ch) < 65536)
        text = ' '.join(text.split())
        
        return text if text else "."
    
    @classmethod
    def _fix_tokenizer(cls, tokenizer):
        """Fix pad_token_id for PhoBERT."""
        if tokenizer.pad_token is None:
            tokenizer.pad_token = "<pad>"
        
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 1
        
        return tokenizer
    
    @classmethod
    def get_scorer(cls, model_type: str = "phobert", device: str = "cuda"):
        """Get or create cached BERTScorer."""
        import bert_score
        from transformers import AutoTokenizer
        
        key = (model_type, device)
        if key in cls._scorers:
            return cls._scorers[key]
        
        model_name = cls.MODEL_MAPPING.get(model_type, model_type)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer = cls._fix_tokenizer(tokenizer)
        
        scorer = bert_score.BERTScorer(
            model_type=model_name,
            num_layers=12,
            batch_size=32,
            nthreads=4,
            all_layers=False,
            idf=False,
            device=device,
            lang=None,
            rescale_with_baseline=False
        )
        
        scorer._tokenizer = tokenizer
        
        cls._scorers[key] = scorer
        return scorer
    
    @classmethod
    def compute_scores(cls, predictions: list, references: list, 
                       model_type: str = "phobert", device: str = "cuda"):
        """Compute BERTScore with basic sanitization."""
        clean_preds = [cls._sanitize_for_bertscore(p) for p in predictions]
        clean_refs = [cls._sanitize_for_bertscore(r) for r in references]
        
        scorer = cls.get_scorer(model_type, device)
        
        P, R, F1 = scorer.score(clean_preds, clean_refs)
        
        return {
            'precision': P.mean().item(),
            'recall': R.mean().item(),
            'f1': F1.mean().item()
        }


# ============================================================================
# SHARED SMILE MODEL
# ============================================================================

class SharedSMILEModel:
    """
    Singleton for SMILE metric model.
    
    SMILE (Sentence-level Metrics for Information-Leveraging Evaluation)
    is used for evaluating answer quality in VQA tasks.
    """
    
    _instances = {}
    
    @classmethod
    def get_instance(cls, model_type: str = 'bert'):
        """Get or initialize shared SMILE model."""
        if model_type not in cls._instances:
            cls._instances[model_type] = SMILE(
                emb_model=model_type,
                eval_metrics=['avg', 'hm'],
                assign_bins=False,
                use_exact_matching=True,
                verbose=False
            )
        return cls._instances[model_type]


# ============================================================================
# SHARED SYNTHETIC ANSWER GENERATOR
# ============================================================================

class SharedSyntheticAnswerGenerator:
    """
    Singleton for Synthetic Answer LLM model.
    
    Uses Qwen model to generate full-sentence synthetic answers
    from question-answer pairs for improved SMILE metric evaluation.
    """
    
    _model = None
    _tokenizer = None
    _device = None
    _initialized = False
    
    # Default model path - can be overridden via environment variable or parameter
    DEFAULT_MODEL_PATH = os.getenv(
        'QWEN_MODEL_PATH',
        '/mnt/dataset1/pretrained_fm/Qwen_Qwen3-8B'
    )
    
    @classmethod
    def initialize(cls, model_path: str = None, device: str = "cuda"):
        """Initialize the synthetic answer generator model."""
        if cls._initialized:
            return
        
        if model_path is None:
            model_path = cls.DEFAULT_MODEL_PATH
        
        from ..utils import load_qwen_text_model
        
        cls._model, cls._tokenizer, cls._device = load_qwen_text_model(
            model_path=model_path,
            device=torch.device(device)
        )
        cls._initialized = True
    
    @classmethod
    def is_initialized(cls) -> bool:
        """Check if the generator has been initialized."""
        return cls._initialized
    
    @classmethod
    def generate_synthetic_answer(cls, question: str, answer: str, max_new_tokens: int = 128) -> str:
        """Generate a synthetic answer for a single question-answer pair."""
        if not cls._initialized:
            raise RuntimeError("SyntheticAnswerGenerator not initialized. Call initialize() first.")
        
        from ..utils import (
            generate_synthetic_answer_qwen_text,
            DEFAULT_SYSTEM_PROMPT,
            DEFAULT_USER_PROMPT_TEMPLATE
        )
        
        return generate_synthetic_answer_qwen_text(
            model=cls._model,
            tokenizer=cls._tokenizer,
            question=question,
            answer=answer,
            device=cls._device,
            max_new_tokens=max_new_tokens,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            user_prompt_template=DEFAULT_USER_PROMPT_TEMPLATE
        )
    
    @classmethod
    def generate_batch(cls, questions: list[str], answers: list[str], 
                       max_new_tokens: int = 128, show_progress: bool = True) -> list[str]:
        """Generate synthetic answers for a batch of question-answer pairs."""
        if not cls._initialized:
            raise RuntimeError("SyntheticAnswerGenerator not initialized. Call initialize() first.")
        
        synthetic_answers = []
        iterator = zip(questions, answers)
        if show_progress:
            iterator = tqdm(list(iterator), desc="Generating synthetic answers")
        
        for q, a in iterator:
            try:
                syn_ans = cls.generate_synthetic_answer(q, a, max_new_tokens)
            except Exception:
                syn_ans = a
            synthetic_answers.append(syn_ans)
        
        return synthetic_answers


__all__ = [
    "SharedBERTScoreModel",
    "SharedSMILEModel", 
    "SharedSyntheticAnswerGenerator",
]


