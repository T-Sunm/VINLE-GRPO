"""
Main dataset loader for GRPO training with ViVQA-X.

This module provides the main entry point for loading and converting ViVQA-X dataset
to MS-Swift GRPO format.
"""

import os
import json
from typing import Optional, Literal


# System prompts for different training modes
SYSTEM_PROMPTS = {
    "grpo": """<image> Bạn là một trợ lý ngôn ngữ thị giác hữu ích, được thiết kế cho suy luận có cấu trúc.""".strip(),
    "sft": """<image> Bạn là một trợ lý ngôn ngữ thị giác hữu ích, được thiết kế cho suy luận có cấu trúc.""".strip(),
    "answer_explain": """<image> Bạn là một trợ lý ngôn ngữ thị giác hữu ích, được thiết kế cho suy luận có cấu trúc.""".strip(),
    "think_answer": """<image> Bạn là một trợ lý ngôn ngữ thị giác hữu ích, được thiết kế cho suy luận có cấu trúc.""".strip(),
}


# User content templates
USER_CONTENT_TEMPLATES = {
    # GRPO: 3 stages (REASONING + CONCLUSION + EXPLANATION)
    "grpo": """Khi trả lời các câu hỏi về hình ảnh, bạn phải trả lời chính xác trong ba giai đoạn, mỗi giai đoạn bắt buộc phải tuân theo format:
<REASONING>[Đưa ra phân tích lập luận chi tiết, từng bước để giải quyết vấn đề.]</REASONING>
<CONCLUSION>[Nêu câu trả lời cuối cùng là một từ hoặc cụm từ.]</CONCLUSION>
<EXPLANATION>[Tổng hợp các thông tin từ REASONING và cho ra câu mô tả ngắn gọn các phân tích đặc điểm.] Hình ảnh cho thấy...</EXPLANATION>
    
Vui lòng áp dụng định dạng này một cách tỉ mỉ để phân tích hình ảnh được cung cấp và trả lời câu hỏi: {question}
Câu trả lời:""".strip(),
    
    # SFT: 2 stages (CONCLUSION + EXPLANATION)
    "sft": """Khi trả lời các câu hỏi về hình ảnh, bạn phải trả lời chính xác trong hai giai đoạn, mỗi giai đoạn bắt buộc phải tuân theo format:
<CONCLUSION>[Nêu câu trả lời cuối cùng là một từ hoặc cụm từ.]</CONCLUSION>
<EXPLANATION>[Giải thích một câu ngắn gọn chứng minh câu trả lời.] Hình ảnh cho thấy...</EXPLANATION>

Vui lòng áp dụng định dạng này một cách tỉ mỉ để phân tích hình ảnh được cung cấp và trả lời câu hỏi: {question}
Câu trả lời:""".strip(),
    
    # Ablation: Only CONCLUSION + EXPLANATION (same as SFT)
    "answer_explain": """Khi trả lời các câu hỏi về hình ảnh, bạn phải trả lời chính xác trong hai giai đoạn, mỗi giai đoạn bắt buộc phải tuân theo format:
<CONCLUSION>[Nêu câu trả lời cuối cùng là một từ hoặc cụm từ.]</CONCLUSION>
<EXPLANATION>[Giải thích một câu ngắn gọn chứng minh câu trả lời.] Hình ảnh cho thấy...</EXPLANATION>

Vui lòng áp dụng định dạng này một cách tỉ mỉ để phân tích hình ảnh được cung cấp và trả lời câu hỏi: {question}
Câu trả lời:""".strip(),
    
    # Ablation: Only REASONING + CONCLUSION (no explanation)
    "think_answer": """Khi trả lời các câu hỏi về hình ảnh, bạn phải trả lời chính xác trong hai giai đoạn, mỗi giai đoạn bắt buộc phải tuân theo format:
<REASONING>[Đưa ra phân tích lập luận chi tiết, từng bước để giải quyết vấn đề.]</REASONING>
<CONCLUSION>[Nêu câu trả lời cuối cùng là một từ hoặc cụm từ.]</CONCLUSION>

Vui lòng áp dụng định dạng này một cách tỉ mỉ để phân tích hình ảnh được cung cấp và trả lời câu hỏi: {question}
Câu trả lời:""".strip(),
}


def load_vivqa_for_grpo(
    split: Literal["train", "val", "test"] = "train",
    mode: Literal["grpo", "sft", "answer_explain", "think_answer"] = "grpo",
    output_file: Optional[str] = None,
    data_dir: str = "/mnt/VLAI_data/ViVQA-X",
    image_base_dir: str = "/mnt/VLAI_data/COCO_Images"
) -> str:
    """
    Main entry point for loading ViVQA-X dataset and converting to GRPO/SFT format.
    
    Args:
        split: Dataset split ('train', 'val', or 'test')
        mode: Training mode:
            - 'grpo': 3 stages (REASONING + CONCLUSION + EXPLANATION)
            - 'sft': 2 stages (CONCLUSION + EXPLANATION)  
            - 'answer_explain': Ablation - 2 stages (CONCLUSION + EXPLANATION)
            - 'think_answer': Ablation - 2 stages (REASONING + CONCLUSION)
        output_file: Custom output file path (optional)
        data_dir: Base directory for ViVQA-X dataset
        image_base_dir: Base directory for COCO images
        
    Returns:
        Path to the created JSONL file
        
    Example:
        >>> load_vivqa_for_grpo(split='train', mode='grpo')
        'data/processed/grpo/ViVQA-X_train_grpo.jsonl'
    """
    if mode not in SYSTEM_PROMPTS:
        raise ValueError(f"Invalid mode: {mode}. Available modes are: {list(SYSTEM_PROMPTS.keys())}")
    
    # Get system prompt and user template
    system_prompt = SYSTEM_PROMPTS[mode]
    user_template = USER_CONTENT_TEMPLATES[mode]
    
    # Determine data path and image directory
    if split == 'train':
        data_path = os.path.join(data_dir, 'ViVQA-X_train.json')
        image_dir = 'train2014'
    elif split == 'val':
        data_path = os.path.join(data_dir, 'ViVQA-X_val.json')
        image_dir = 'val2014'
    else:  # test
        data_path = os.path.join(data_dir, 'ViVQA-X_test.json')
        image_dir = 'val2014'
    
    # Load raw data
    print(f"Loading data from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Determine output file
    if output_file is None:
        output_dir = f'data/processed/{mode}'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'ViVQA-X_{split}_{mode}.jsonl')
    
    # Convert and write data
    print(f"Writing to: {output_file}")
    count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for idx, item in enumerate(raw_data):
            image_name = item.get('image_name')
            question = item.get('question')
            answer = item.get('answer')
            explanations = item.get('explanation')
            
            # Validate required fields
            if not all([image_name, question, answer, explanations]):
                continue
            
            # Extract explanation
            if isinstance(explanations, list) and len(explanations) > 0:
                explanation = explanations[0]
            elif isinstance(explanations, str):
                explanation = explanations
            else:
                continue
            
            # Construct absolute image path
            absolute_image_path = os.path.join(image_base_dir, image_dir, image_name)
            
            # Format user content
            user_content = user_template.format(question=question)
            
            # Format assistant response (solution) based on mode
            if mode in ["grpo", "sft", "answer_explain"]:
                # Modes with CONCLUSION + EXPLANATION
                solution = f"<CONCLUSION>{answer}</CONCLUSION>\n<EXPLANATION>{explanation}</EXPLANATION>"
            elif mode == "think_answer":
                # Mode with REASONING + CONCLUSION (use explanation as reasoning)
                solution = f"<REASONING>{explanation}</REASONING>\n<CONCLUSION>{answer}</CONCLUSION>"
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            # MS-Swift format
            entry = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "images": [absolute_image_path],
                "solution": solution  # For GRPO training
            }
            
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
            count += 1
    
    print(f"✅ Created {output_file} with {count} entries")
    return output_file


def main():
    """CLI entry point for data preparation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare ViVQA-X dataset for GRPO/SFT training')
    parser.add_argument('--split', type=str, default='all', choices=['train', 'val', 'test', 'all'],
                        help='Dataset split to process')
    parser.add_argument('--mode', type=str, default='grpo', 
                        choices=['grpo', 'sft', 'answer_explain', 'think_answer'],
                        help='Training mode: grpo (3 stages), sft (2 stages), answer_explain (ablation), think_answer (ablation)')
    parser.add_argument('--data_dir', type=str, default='/mnt/VLAI_data/ViVQA-X',
                        help='Base directory for ViVQA-X dataset')
    parser.add_argument('--image_dir', type=str, default='/mnt/VLAI_data/COCO_Images',
                        help='Base directory for COCO images')
    
    args = parser.parse_args()
    
    # Process splits
    splits = ['train', 'val', 'test'] if args.split == 'all' else [args.split]
    
    print(f"=== Preparing ViVQA-X dataset for {args.mode.upper()} training ===\n")
    
    for split in splits:
        print(f"--- Processing {split} split ---")
        load_vivqa_for_grpo(
            split=split,
            mode=args.mode,
            data_dir=args.data_dir,
            image_base_dir=args.image_dir
        )
        print()


if __name__ == "__main__":
    main()
