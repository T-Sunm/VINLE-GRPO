"""
OEA Ablation Inference for InternVL/Vintern
Tags: CONCLUSION + EXPLANATION (no REASONING)
"""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm

from ..other_models.internvl import InternVLModel
from ..common import get_oea_prompt, parse_oea_output, process_inference_sample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer(model: InternVLModel, question: str, image_path: str) -> dict:
    """Run OEA inference."""
    pixel_values = model._load_image(image_path).to(torch.bfloat16).to(device)
    prompt = get_oea_prompt(question)
    
    with torch.no_grad():
        response = model.model.chat(
            model.tokenizer,
            pixel_values,
            prompt,
            generation_config={"max_new_tokens": 800, "pad_token_id": model.tokenizer.eos_token_id}
        )
    return parse_oea_output(response)


def main():
    parser = argparse.ArgumentParser(description="OEA Inference")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data_path", default="/mnt/VLAI_data/ViVQA-X/ViVQA-X_test.json")
    parser.add_argument("--image_folder", default="/mnt/VLAI_data/COCO_Images/val2014")
    parser.add_argument("--output_dir", default="results/inference/oea")
    parser.add_argument("--output_name", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("OEA Inference (CONCLUSION + EXPLANATION)")
    print("=" * 80)
    
    model = InternVLModel()
    model.model_path = args.model
    
    with open(args.data_path, "r") as f:
        data = json.load(f)
    data = list(data.values()) if isinstance(data, dict) else data
    if args.limit:
        data = data[:args.limit]
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.output_name or Path(args.model).name}.json"
    
    image_folder = Path(args.image_folder)
    for item in tqdm(data, desc="OEA Inference"):
        process_inference_sample(model, item, image_folder, infer)
    
    with open(output_file, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved to: {output_file}")


if __name__ == "__main__":
    main()
