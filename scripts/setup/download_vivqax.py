#!/usr/bin/env python3
"""
Download ViVQA-X dataset from Hugging Face
Dataset: https://huggingface.co/datasets/VLAI-AIVN/ViVQA-X
Output: data/raw/vivqa-x/annotations/
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any

try:
    from datasets import load_dataset
    from tqdm import tqdm
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call(["pip", "install", "datasets", "tqdm"])
    from datasets import load_dataset
    from tqdm import tqdm


class ViVQAXDownloader:
    """Download and save ViVQA-X dataset from Hugging Face"""
    
    def __init__(self, output_dir: str = None):
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        
        if output_dir is None:
            self.output_dir = project_root / "data" / "raw" / "vivqa-x" / "annotations"
        else:
            self.output_dir = Path(output_dir)
        
        self.dataset_name = "VLAI-AIVN/ViVQA-X"
        self.splits = ["train", "val", "test"]
        
    def create_directories(self):
        """Create necessary directories"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")
        
    def download_and_save_split(self, split: str) -> Dict[str, Any]:
        """Download a specific split and save to JSON file"""
        print(f"\n[{split.upper()}]")
        
        try:
            dataset = load_dataset(self.dataset_name, split=split)
            
            data = []
            for item in tqdm(dataset, desc=f"Processing"):
                data.append({
                    "question": item["question"],
                    "image_id": item["image_id"],
                    "image_name": item["image_name"],
                    "explanation": item["explanation"],
                    "answer": item["answer"],
                    "question_id": item["question_id"]
                })
            
            output_filename = f"ViVQA-X_{split}.json"
            output_path = self.output_dir / output_filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"Saved {len(data):,} samples ({file_size_mb:.2f} MB)")
            
            return {
                "split": split,
                "num_samples": len(data),
                "output_file": str(output_path),
                "file_size_mb": file_size_mb
            }
            
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def show_sample(self, split: str = "train", num_samples: int = 1):
        """Show sample data from a split"""
        file_path = self.output_dir / f"ViVQA-X_{split}.json"
        
        if not file_path.exists():
            return
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\nSample from {split}:")
        for i, sample in enumerate(data[:num_samples]):
            print(f"  Question: {sample['question']}")
            print(f"  Answer: {sample['answer']}")
            print(f"  Image: {sample['image_name']}")
            print(f"  Explanations: {len(sample['explanation'])}")
    
    def run(self):
        """Main execution"""
        print("ViVQA-X Dataset Downloader")
        print("=" * 50)
        
        self.create_directories()
        
        all_stats = []
        for split in self.splits:
            stats = self.download_and_save_split(split)
            if stats:
                all_stats.append(stats)
        
        print("\n" + "=" * 50)
        print("Summary:")
        total_samples = 0
        total_size = 0
        for stats in all_stats:
            print(f"  {stats['split']:12s}: {stats['num_samples']:,} samples "
                  f"({stats['file_size_mb']:.2f} MB)")
            total_samples += stats['num_samples']
            total_size += stats['file_size_mb']
        
        print(f"  Total: {total_samples:,} samples ({total_size:.2f} MB)")
        
        if all_stats:
            self.show_sample("train", num_samples=1)
        
        print("\nNote: Images are from COCO 2014 dataset")
        print("Download with: bash scripts/download_data.sh")


def main():
    downloader = ViVQAXDownloader()
    downloader.run()


if __name__ == "__main__":
    main()