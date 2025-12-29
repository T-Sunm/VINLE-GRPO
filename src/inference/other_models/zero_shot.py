"""
Zero-shot Inference Runner for Other VLM Models

Supports: QwenVL, Molmo, Phi, Ovis, MiniCPM, VideoLLaMA, InternVL, Vintern1B
All models use GRPO prompt for structured comparison.
"""

import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.common import get_grpo_prompt, parse_grpo_output


# Model registry
MODELS = {
    "internvl": "other_models.internvl.InternVLModel",
    "vintern1b": "other_models.vintern1b.Vintern1BModel",
    "qwenvl": "other_models.qwenvl.QwenVLModel",
    "molmo": "other_models.molmo.MolmoModel",
    "phi": "other_models.phi.PhiModel",
    "ovis": "other_models.ovis.OvisModel",
    "minicpm": "other_models.minicpm.MiniCPMModel",
    "videollama": "other_models.videollama.VideoLLaMAModel",
}


def import_model_class(model_key: str):
    """Dynamically import model class."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")
    
    module_path, class_name = MODELS[model_key].rsplit(".", 1)
    print(f"📦 Importing {class_name}...")
    
    # Import from other_models directory
    import importlib
    module = importlib.import_module(f"src.inference.{module_path}")
    return getattr(module, class_name)


def process_sample(model, item: dict, image_folder: Path) -> dict:
    """Process single sample with GRPO prompt."""
    if "image_name" not in item and "image_id" in item:
        item["image_name"] = f"COCO_val2014_{int(item['image_id']):012d}.jpg"
    
    if "answer" not in item and "answers" in item:
        item["answer"] = item["answers"][0]["answer"]
    
    img_path = image_folder / item["image_name"]
    
    if not img_path.exists():
        print(f"⚠️  Image not found: {img_path}")
        item["predict"] = "ERROR: Image file not found"
        return item
    
    try:
        # Use GRPO inference method (all models have infer_grpo)
        reasoning, answer, explanation = model.infer_grpo(item["question"], str(img_path))
        
        item["thinking"] = reasoning
        item["predict"] = answer
        item["pred_explanation"] = explanation
        
    except Exception as e:
        print(f"❌ Error processing: {e}")
        item["predict"] = f"ERROR: {str(e)}"
    
    return item


def main():
    parser = argparse.ArgumentParser(description="Zero-shot Inference for Other VLMs")
    parser.add_argument("model", choices=MODELS.keys(), help="Model to run")
    parser.add_argument("--image_folder", default="/mnt/VLAI_data/COCO_Images/val2014")
    parser.add_argument("--data_path", default="/mnt/VLAI_data/ViVQA-X/ViVQA-X_test.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/inference/zeroshot")
    parser.add_argument("--output_name", default=None)
    parser.add_argument("--limit", type=int, default=None)
    
    args = parser.parse_args()
    
    # Set seed
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("=" * 80)
    print(f"Zero-shot Inference: {args.model.upper()}")
    print("=" * 80)
    
    # Load model
    print(f"🚀 Initializing {args.model}...")
    try:
        ModelClass = import_model_class(args.model)
        model = ModelClass()
        print(f"✅ Loaded {model.model_name}\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 1
    
    # Load data
    print(f"📂 Loading data from {args.data_path}...")
    with open(args.data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    data = list(raw_data.values()) if isinstance(raw_data, dict) else raw_data
    if args.limit:
        data = data[:args.limit]
    
    print(f"📝 Processing {len(data)} samples...\n")
    
    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or model.model_name
    output_file = output_dir / f"{output_name}.json"
    
    print(f"💾 Results will be saved to: {output_file}\n")
    
    # Process samples
    image_folder = Path(args.image_folder)
    for item in tqdm(data, desc=f"{args.model} inference"):
        try:
            process_sample(model, item, image_folder)
        except Exception as e:
            print(f"❌ Error: {e}")
            item["pred_answer"] = f"ERROR: {e}"
    
    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Saved to: {output_file}")
    print("🎉 Done!")
    return 0


if __name__ == "__main__":
    exit(main())
