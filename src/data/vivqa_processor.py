"""
ViVQA-X dataset processor.

This module handles ViVQA-X specific processing logic, including:
- Dataset loading and validation
- Format-specific conversions
- Question-answer pair processing
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ViVQAXSample:
    """Data class for a single ViVQA-X sample."""
    image_name: str
    image_id: str
    question: str
    answer: str
    explanation: str
    split: str = "train"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'image_name': self.image_name,
            'image_id': self.image_id,
            'question': self.question,
            'answer': self.answer,
            'explanation': self.explanation,
            'split': self.split
        }


class ViVQAProcessor:
    """Processor for ViVQA-X dataset."""
    
    def __init__(
        self,
        data_dir: str = "/mnt/VLAI_data/ViVQA-X",
        image_base_dir: str = "/mnt/VLAI_data/COCO_Images"
    ):
        """
        Initialize ViVQA-X processor.
        
        Args:
            data_dir: Base directory for ViVQA-X dataset
            image_base_dir: Base directory for COCO images
        """
        self.data_dir = data_dir
        self.image_base_dir = image_base_dir
        
        # Split to image directory mapping
        self.split_to_image_dir = {
            'train': 'train2014',
            'val': 'val2014',
            'test': 'val2014'
        }
    
    def load_split(self, split: str) -> List[Dict]:
        """
        Load a specific split of ViVQA-X dataset.
        
        Args:
            split: Dataset split ('train', 'val', or 'test')
            
        Returns:
            List of dataset samples
        """
        if split not in self.split_to_image_dir:
            raise ValueError(f"Invalid split: {split}. Must be one of {list(self.split_to_image_dir.keys())}")
        
        data_path = os.path.join(self.data_dir, f'ViVQA-X_{split}.json')
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def parse_sample(self, item: Dict, split: str) -> Optional[ViVQAXSample]:
        """
        Parse a single ViVQA-X sample.
        
        Args:
            item: Raw data item from dataset
            split: Dataset split
            
        Returns:
            Parsed ViVQAXSample or None if invalid
        """
        image_name = item.get('image_name')
        image_id = item.get('image_id')
        question = item.get('question')
        answer = item.get('answer')
        explanations = item.get('explanation')
        
        # Validate required fields
        if not all([image_name, question, answer, explanations]):
            return None
        
        # Extract explanation
        if isinstance(explanations, list) and len(explanations) > 0:
            explanation = explanations[0]
        elif isinstance(explanations, str):
            explanation = explanations
        else:
            return None
        
        return ViVQAXSample(
            image_name=image_name,
            image_id=str(image_id) if image_id else "",
            question=question,
            answer=answer,
            explanation=explanation,
            split=split
        )
    
    def get_image_path(self, image_name: str, split: str) -> str:
        """
        Get absolute path to image file.
        
        Args:
            image_name: Name of the image file
            split: Dataset split
            
        Returns:
            Absolute path to image file
        """
        image_dir = self.split_to_image_dir[split]
        return os.path.join(self.image_base_dir, image_dir, image_name)
    
    def get_statistics(self, split: str) -> Dict:
        """
        Get statistics for a dataset split.
        
        Args:
            split: Dataset split
            
        Returns:
            Dictionary with statistics
        """
        data = self.load_split(split)
        
        stats = {
            'total_samples': len(data),
            'valid_samples': 0,
            'answer_types': {},
            'avg_question_length': 0,
            'avg_answer_length': 0,
            'avg_explanation_length': 0
        }
        
        total_q_len = 0
        total_a_len = 0
        total_e_len = 0
        
        for item in data:
            sample = self.parse_sample(item, split)
            if sample is None:
                continue
            
            stats['valid_samples'] += 1
            
            # Count answer types (yes/no, number, other)
            answer_lower = sample.answer.lower().strip()
            if answer_lower in ['có', 'không', 'yes', 'no']:
                answer_type = 'yes/no'
            elif answer_lower.isdigit():
                answer_type = 'number'
            else:
                answer_type = 'other'
            
            stats['answer_types'][answer_type] = stats['answer_types'].get(answer_type, 0) + 1
            
            # Accumulate lengths
            total_q_len += len(sample.question.split())
            total_a_len += len(sample.answer.split())
            total_e_len += len(sample.explanation.split())
        
        # Calculate averages
        if stats['valid_samples'] > 0:
            stats['avg_question_length'] = total_q_len / stats['valid_samples']
            stats['avg_answer_length'] = total_a_len / stats['valid_samples']
            stats['avg_explanation_length'] = total_e_len / stats['valid_samples']
        
        return stats
    
    def print_statistics(self, split: str) -> None:
        """
        Print dataset statistics.
        
        Args:
            split: Dataset split
        """
        stats = self.get_statistics(split)
        
        print(f"\n{'='*60}")
        print(f"ViVQA-X {split.upper()} Split Statistics")
        print(f"{'='*60}")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Valid samples: {stats['valid_samples']}")
        print(f"\nAnswer type distribution:")
        for answer_type, count in sorted(stats['answer_types'].items()):
            percentage = (count / stats['valid_samples']) * 100
            print(f"  - {answer_type}: {count} ({percentage:.1f}%)")
        print(f"\nAverage lengths (words):")
        print(f"  - Question: {stats['avg_question_length']:.1f}")
        print(f"  - Answer: {stats['avg_answer_length']:.1f}")
        print(f"  - Explanation: {stats['avg_explanation_length']:.1f}")
        print(f"{'='*60}\n")


def main():
    """CLI entry point for ViVQA-X processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process ViVQA-X dataset')
    parser.add_argument('--split', type=str, default='all', choices=['train', 'val', 'test', 'all'],
                        help='Dataset split to analyze')
    parser.add_argument('--data_dir', type=str, default='/mnt/VLAI_data/ViVQA-X',
                        help='Base directory for ViVQA-X dataset')
    parser.add_argument('--image_dir', type=str, default='/mnt/VLAI_data/COCO_Images',
                        help='Base directory for COCO images')
    
    args = parser.parse_args()
    
    processor = ViVQAProcessor(data_dir=args.data_dir, image_base_dir=args.image_dir)
    
    splits = ['train', 'val', 'test'] if args.split == 'all' else [args.split]
    
    for split in splits:
        processor.print_statistics(split)


if __name__ == "__main__":
    main()
