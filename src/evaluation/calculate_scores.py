"""
Unified VQA Evaluation Script for VINLE-GRPO.

Supports all inference output formats:
- GRPO: thinking + pred_explanation + predict
- OEA: pred_explanation + predict
- OTA: thinking + predict
- ZEROSHOT: thinking + pred_explanation + predict
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
    validate_format_consistency,
    normalize_explanation,
    ensure_list,
    SharedBERTScoreModel,
    SharedSMILEModel,
    SharedSyntheticAnswerGenerator,
)

from .metrics import (
    compute_accuracy,
    get_nlg_scores,
    compute_smile_scores,
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_explanation_field(item: Dict, format_info: Dict) -> str:
    """
    Extract explanation field with fallback logic.
    
    Priority: pred_explanation > thinking > empty
    """
    if format_info['has_pred_explanation'] and item.get("pred_explanation", "").strip():
        return normalize_explanation(item.get("pred_explanation", ""))
    elif format_info['has_thinking']:
        return normalize_explanation(item.get("thinking", ""))
    return ""


def add_scores_to_row(row: Dict, scores: Dict, prefix: str = "") -> None:
    """Add scores to row dictionary with optional prefix."""
    if not scores:
        return
    
    for key, value in scores.items():
        score_key = f"{prefix}_{key}" if prefix else key
        row[score_key] = round(value, 2)


# ============================================================================
# FILE EVALUATION
# ============================================================================

def evaluate_file(json_path: str, device: str = "cuda") -> Dict:
    """
    Evaluate a single inference output file.
    
    Auto-detects format and computes applicable metrics.
    """
    print(f"Loading: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Detect format
    format_info = detect_format(data)
    print(f"Format detected: {format_info['format_name']}")
    
    if not validate_format_consistency(data):
        print("Warning: Inconsistent format detected!")
    
    # Compute accuracy
    print("Computing accuracy...")
    accuracy_results = compute_accuracy(data)
    
    # Prepare data
    all_gt_expls = []
    all_questions = []
    all_gt_answers = []
    all_pred_answers = []
    by_type = {}
    
    for item in data:
        gt_expls = [normalize_explanation(e) for e in ensure_list(item.get("explanation", []))]
        all_gt_expls.append(gt_expls)
        all_questions.append(item.get("question", ""))
        all_gt_answers.append(item.get("answer", ""))
        all_pred_answers.append(item.get("predict", ""))
        
        ans_type = item.get("answer_type", "other")
        if ans_type not in by_type:
            by_type[ans_type] = {
                "gt_expls": [],
                "pred_fields": [],
                "questions": [],
                "gt_answers": [],
                "pred_answers": [],
            }
        
        by_type[ans_type]["gt_expls"].append(gt_expls)
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
    
    # Evaluate explanation field (pred_explanation or thinking)
    if format_info['has_pred_explanation'] or format_info['has_thinking']:
        print("Evaluating explanations...")
        
        # Extract explanations with fallback logic
        all_explanations = [get_explanation_field(item, format_info) for item in data]
        
        explanation_scores = get_nlg_scores(all_gt_expls, all_explanations, device, model_type='phobert')
        results['explanation_scores'] = explanation_scores
        
        # Collect by type
        for ans_type in by_type:
            type_expls = [
                get_explanation_field(item, format_info)
                for item in data if item.get("answer_type", "other") == ans_type
            ]
            by_type[ans_type]["pred_fields"] = type_expls
    
    # Evaluate answers with SMILE
    print("Computing SMILE scores...")
    
    if not SharedSyntheticAnswerGenerator.is_initialized():
        SharedSyntheticAnswerGenerator.initialize(device=device)
    
    all_synthetic_answers = SharedSyntheticAnswerGenerator.generate_batch(
        questions=all_questions,
        answers=all_gt_answers,
        max_new_tokens=128,
        show_progress=True
    )
    
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
        
        # Explanation scores
        if type_data.get('pred_fields'):
            nlg_scores_type = get_nlg_scores(
                type_data['gt_expls'],
                type_data['pred_fields'],
                device,
                model_type='phobert'
            )
            type_results['explanation_scores'] = nlg_scores_type
        
        # Answer scores
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
    """Format evaluation results into DataFrame rows."""
    rows = []
    
    # Overall row
    overall_row = {
        'model': model_name,
        'answer_type': 'Overall',
        'total': results['total_examples'],
        'correct': results['correct_count'],
        'accuracy': round(results['accuracy'], 2),
    }
    add_scores_to_row(overall_row, results.get('explanation_scores'), 'explanation')
    add_scores_to_row(overall_row, results.get('answer_scores'))
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
        add_scores_to_row(type_row, type_data.get('explanation_scores'), 'explanation')
        add_scores_to_row(type_row, type_data.get('answer_scores'))
        rows.append(type_row)
    
    return rows


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified VQA Evaluation for VINLE-GRPO")
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
    print("VINLE-GRPO VQA Evaluation")
    print(f"{'='*80}")
    print(f"Input: {args.input_dir}")
    print(f"Files: {len(file_paths)}")
    print(f"Device: {args.device}")
    print(f"{'='*80}\n")
    
    # Initialize shared models
    print("Initializing models...")
    SharedBERTScoreModel.get_scorer(model_type='phobert', device='cpu')
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
            results = evaluate_file(file_path, device=args.device)
            rows = format_results_to_dataframe(results, model_name)
            all_rows.extend(rows)
            print(f"Done - Accuracy: {results['accuracy']:.2f}%")
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
        csv_path = os.path.join(args.input_dir, f"evaluation_results_{timestamp}.csv")
    
    df.to_csv(csv_path, index=False, encoding="utf-8")
    
    print(f"\n{'='*80}")
    print("Evaluation completed!")
    print(f"Results saved to: {csv_path}")
    print(f"{'='*80}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
