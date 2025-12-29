import os
import json
import re # Import re for the reward function
from typing import List # Import List for type hinting
from datetime import datetime # Import datetime for logging

# --- Dataset Loader Function ---

# system_prompt = """You are a Visual Question Answering system. Your task is to answer and explain questions based on the content of the provided image.
# Follow these steps precisely:
#     - <think> Your step-by-step reasoning process. Analyze the image carefully. </think>
#     - <answer> Based on your reasoning, provide your Vietnamese answer must be one word or a short phrase. </answer>
#     - <explain> Your brief Vietnamese explanation, it is one sentence that justifies your answer. </explain>""".strip()

# user_prompt_template = """<image>
# Question: {question}""".strip()

# System prompt hoàn toàn bằng tiếng Việt, ra chỉ thị rõ ràng
system_prompt = """Bạn là một hệ thống Hỏi-Đáp Trực quan (VQA). 
Nhiệm vụ của bạn là trả lời và giải thích các câu hỏi dựa trên nội dung hình ảnh.
Hãy tuân thủ chính xác các bước và định dạng thẻ sau:
1. <think> Quá trình suy luận từng bước của bạn. Phân tích kỹ hình ảnh và câu hỏi. </think>
2. <explain> Lời giải thích ngắn gọn bằng tiếng Việt (thường là một câu) để chứng minh cho câu trả lời. </explain>
3. <answer> Dựa trên suy luận, đưa ra câu trả lời cuối cùng. Câu trả lời phải là tiếng Việt, là một từ hoặc cụm từ ngắn. </answer>""".strip()

# User prompt bâyT giờ rất sạch sẽ, chỉ chứa dữ liệu
user_prompt_template = """<image>
Câu hỏi: {question}""".strip()


def create_jsonl_for_swift_vlm_grpo(split="train", output_file=None, image_base_dir="/mnt/VLAI_data/COCO_Images"):
    """
    Tạo file JSONL theo format GRPO cho ms-swift VLM.
    Chứa system prompt, user prompt và các cột cần thiết.
    """
    data_dir = "/mnt/VLAI_data/ViVQA-X" # Directory containing the original JSON files

    # Determine input JSON path based on split
    if split == 'train':
        data_path = os.path.join(data_dir, 'ViVQA-X_train.json')
        image_sub_dir = 'train2014'
    elif split == 'val':
        data_path = os.path.join(data_dir, 'ViVQA-X_val.json')
        image_sub_dir = 'val2014'
    else: # Assuming 'test' split or similar for validation
        data_path = os.path.join(data_dir, 'ViVQA-X_val.json') # Hoặc ViVQA-X_test.json tùy bạn đặt tên
        image_sub_dir = 'val2014'

    # Load the original VQA data
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {data_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {data_path}")
        return None

    # Determine output file path
    if output_file is None:
        output_dir = 'data_custom' # Output to data_custom/ in current directory
        os.makedirs(output_dir, exist_ok=True)
        output_file_split_name = split if split != 'val' else 'val'
        output_file = os.path.join(output_dir, f'ViVQA-X_{output_file_split_name}_grpo.jsonl')


    print(f"Processing GRPO split: {split}")
    print(f"Reading from: {data_path}")
    print(f"Writing to: {output_file}")

    processed_count = 0
    skipped_count = 0
    # Process and write data in jsonl format
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in raw_data:
            image_name = item.get('image_name')
            question = item.get('question')
            answer = item.get('answer')
            explanations = item.get('explanation') # Assuming it's a list

            # Basic validation
            if not all([image_name, question, answer]) or not explanations or not explanations[0]:
                skipped_count += 1
                continue

            # Construct absolute image path
            absolute_image_path = os.path.join(image_base_dir, image_sub_dir, image_name)

            explanation = explanations[0] # Take the first explanation

            # Format user prompt
            user_content = user_prompt_template.format(question=question)

            # Define solution string
            assistant_ground_truth = f"<think></think><answer>{answer}</answer><explain>{explanation}</explain>"

            # --- MODIFIED: Create the entry for GRPO with separate system message ---
            swift_entry = {
                "messages": [
                    {"role": "system", "content": system_prompt}, # Thêm system message
                    {"role": "user", "content": user_content}
                ],
                "images": [absolute_image_path],
                "solution": assistant_ground_truth
            }
            # --- END MODIFIED ---

            f_out.write(json.dumps(swift_entry, ensure_ascii=False) + '\n')
            processed_count += 1

    print(f"Finished processing GRPO {split} split.")
    print(f"Processed items: {processed_count}")
    print(f"Skipped items: {skipped_count}")
    print(f"Output saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    # --- Configuration ---
    IMAGE_BASE_DIRECTORY = "/mnt/VLAI_data/COCO_Images"
    # --- End Configuration ---

    print("Starting GRPO dataset conversion...")
    create_jsonl_for_swift_vlm_grpo("train", image_base_dir=IMAGE_BASE_DIRECTORY)
    create_jsonl_for_swift_vlm_grpo("val", image_base_dir=IMAGE_BASE_DIRECTORY)
    # create_jsonl_for_swift_vlm_grpo("test", image_base_dir=IMAGE_BASE_DIRECTORY)
    print("GRPO dataset conversion finished.")