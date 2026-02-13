"""
VQA Evaluation Script - SMILE Metrics.

Computes: SMILE scores using synthetic answers from LLM.
Run in vqa-nle-smile env (has flash-attention).

Usage:
    python -m src.evaluation.calculate_smile_scores --input-dir outputs/inference/zeroshot
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
import argparse
import pandas as pd
from datetime import datetime
from typing import Dict, List

from .core import (
    detect_format,
    normalize_answer,
)

from .core.shared_models import (
    SharedSyntheticAnswerGenerator,
    SharedSMILEModel,
)

from .metrics import (
    compute_accuracy,
    compute_smile_scores,
)


# ============================================================================
# FILE EVALUATION
# ============================================================================

def evaluate_smile(json_path: str, device: str = "cuda") -> Dict:
    """
    Evaluate SMILE scores for a single inference output file.
    """
    print(f"Loading: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Detect format
    format_info = detect_format(data)   
    print(f"Format detected: {format_info['format_name']}")
    
    # Compute accuracy (for reference)
    accuracy_results = compute_accuracy(data)
    
    # Prepare data
    all_questions = []
    all_gt_answers = []
    all_pred_answers = []
    by_type = {}
    
    for item in data:
        all_questions.append(item.get("question", ""))
        all_gt_answers.append(item.get("answer", ""))
        all_pred_answers.append(item.get("predict", ""))
        
        ans_type = item.get("answer_type", "other")
        if ans_type not in by_type:
            by_type[ans_type] = {
                "questions": [],
                "gt_answers": [],
                "pred_answers": [],
            }
        
        by_type[ans_type]["questions"].append(item.get("question", ""))
        by_type[ans_type]["gt_answers"].append(item.get("answer", ""))
        by_type[ans_type]["pred_answers"].append(item.get("predict", ""))
    
    # Initialize results
    results = {
        "format": format_info['format_name'],
        "total_examples": accuracy_results['total'],
        "correct_count": accuracy_results['correct'],
        "accuracy": accuracy_results['accuracy'],
    }
    
    # Generate synthetic answers
    print("Generating synthetic answers...")
    if not SharedSyntheticAnswerGenerator.is_initialized():
        SharedSyntheticAnswerGenerator.initialize(device=device)
    
    all_synthetic_answers = SharedSyntheticAnswerGenerator.generate_batch(
        questions=all_questions,
        answers=all_gt_answers,
        max_new_tokens=128,
        show_progress=True
    )
    
    # Compute SMILE scores
    print("Computing SMILE scores...")
    smile_scores = compute_smile_scores(
        all_questions,
        all_gt_answers,
        all_pred_answers,
        synthetic_answers=all_synthetic_answers,
        model_type='phobert'
    )
    results['answer_scores'] = smile_scores
    
    # Compute by answer type
    print("Computing by answer type...")
    results['by_answer_type'] = {}
    
    for ans_type, type_data in by_type.items():
        type_accuracy = accuracy_results['by_answer_type'].get(ans_type, {})
        
        type_results = {
            'total_examples': type_accuracy.get('total', 0),
            'correct_count': type_accuracy.get('correct', 0),
            'accuracy': type_accuracy.get('accuracy', 0),
        }
        
        if type_data['questions']:
            synthetic_answers_type = SharedSyntheticAnswerGenerator.generate_batch(
                type_data['questions'],
                type_data['gt_answers'],
                max_new_tokens=128,
                show_progress=False
            )
            
            smile_scores_type = compute_smile_scores(
                type_data['questions'],
                type_data['gt_answers'],
                type_data['pred_answers'],
                synthetic_answers=synthetic_answers_type,
                model_type='phobert'
            )
            type_results['answer_scores'] = smile_scores_type
        
        results['by_answer_type'][ans_type] = type_results
    
    return results


# ============================================================================
# RESULTS FORMATTING
# ============================================================================

def format_results_to_dataframe(results: Dict, model_name: str) -> List[Dict]:
    """Format SMILE results into DataFrame rows."""
    rows = []
    
    # Overall row
    overall_row = {
        'model': model_name,
        'answer_type': 'Overall',
        'total': results['total_examples'],
        'correct': results['correct_count'],
        'accuracy': round(results['accuracy'], 2),
    }
    if results.get('answer_scores'):
        for key, value in results['answer_scores'].items():
            overall_row[key] = round(value, 2)
    rows.append(overall_row)
    
    # By answer type rows
    for ans_type, type_data in results.get('by_answer_type', {}).items():
        type_row = {
            'model': model_name,
            'answer_type': ans_type,
            'total': type_data['total_examples'],
            'correct': type_data['correct_count'],
            'accuracy': round(type_data['accuracy'], 2),
        }
        if type_data.get('answer_scores'):
            for key, value in type_data['answer_scores'].items():
                type_row[key] = round(value, 2)
        rows.append(type_row)
    
    return rows


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="VQA Evaluation - SMILE Metrics")
    parser.add_argument("--input-dir", type=str, default="outputs/inference",
                       help="Directory containing JSON inference results")
    parser.add_argument("--filenames", nargs="+", default=[],
                       help="Specific filenames to evaluate")
    parser.add_argument("--output-file", type=str, default=None,
                       help="Output CSV filename (auto-generated if not specified)")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device for model computation")
    parser.add_argument("--cuda-device", type=str, default="0",
                       help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--bert-device", type=str, default="cpu",
                       help="Device for BERTScore in SMILE (default: cpu to avoid CUDA asserts)")
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    
    # Find files to evaluate
    if args.filenames:
        files = [f if f.endswith(".json") else f"{f}.json" for f in args.filenames]
        file_paths = [os.path.join(args.input_dir, f) for f in files]
    else:
        files = [f for f in os.listdir(args.input_dir)
                if f.endswith(".json") and "_score" not in f and "summary" not in f]
        file_paths = [os.path.join(args.input_dir, f) for f in sorted(files)]
    
    if not file_paths:
        print("No JSON files found!")
        return
    
    # Print header
    print(f"\n{'='*80}")
    print("VINLE-GRPO VQA Evaluation (SMILE Metrics)")
    print(f"{'='*80}")
    print(f"Input: {args.input_dir}")
    print(f"Files: {len(file_paths)}")
    print(f"Device: {args.device}")
    print(f"{'='*80}\n")
    
    # Initialize shared models
    print("Initializing models...")
    # SMILE uses PhoBERT internally - init on specified device
    from .core.shared_models import SharedBERTScoreModel
    SharedBERTScoreModel.get_scorer(model_type='phobert', device=args.bert_device)
    SharedSMILEModel.get_instance(model_type='phobert')
    SharedSyntheticAnswerGenerator.initialize(device=args.device)
    print("Models ready\n")
    
    # Evaluate files
    all_rows = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for file_path in file_paths:
        model_name = os.path.splitext(os.path.basename(file_path))[0]
        
        print(f"\n{'─'*80}")
        print(f"Evaluating: {model_name}")
        print(f"{'─'*80}")
        
        try:
            results = evaluate_smile(file_path, device=args.device)
            rows = format_results_to_dataframe(results, model_name)
            all_rows.extend(rows)
            print(f"Done - SMILE_avg: {results['answer_scores'].get('SMILE_avg', 0):.2f}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    if not all_rows:
        print("\nNo results to save!")
        return
    
    df = pd.DataFrame(all_rows)
    
    if args.output_file:
        csv_path = args.output_file if args.output_file.endswith(".csv") else f"{args.output_file}.csv"
    else:
        csv_path = os.path.join(args.input_dir, f"smile_results_{timestamp}.csv")
    
    df.to_csv(csv_path, index=False, encoding="utf-8")
    
    print(f"\n{'='*80}")
    print("SMILE Evaluation completed!")
    print(f"Results saved to: {csv_path}")
    print(f"{'='*80}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
