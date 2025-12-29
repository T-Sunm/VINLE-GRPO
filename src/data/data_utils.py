"""
Utility functions for data processing and image preprocessing.

This module provides utilities for:
- Image preprocessing and transformation
- Data format conversion
- File path handling
"""

import os
import json
import torch
from PIL import Image
from typing import List, Tuple, Union
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================================
# Image Preprocessing Functions
# ============================================================================

def build_transform(input_size: int) -> T.Compose:
    """
    Build image transformation pipeline.
    
    Args:
        input_size: Target image size
        
    Returns:
        Composed transformations
    """
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int
) -> Tuple[int, int]:
    """
    Find the closest aspect ratio from a list of target ratios.
    
    Args:
        aspect_ratio: Current image aspect ratio
        target_ratios: List of (width_ratio, height_ratio) tuples
        width: Image width
        height: Image height
        image_size: Target image size
        
    Returns:
        Best matching (width_ratio, height_ratio) tuple
    """
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False
) -> List[Image.Image]:
    """
    Dynamically preprocess image with adaptive aspect ratio splitting.
    
    Args:
        image: Input PIL Image
        min_num: Minimum number of image tiles
        max_num: Maximum number of image tiles
        image_size: Size of each tile
        use_thumbnail: Whether to add a thumbnail image
        
    Returns:
        List of processed image tiles
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    
    # Generate target ratios
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    
    # Find best aspect ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    
    # Resize and split image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    # Add thumbnail if needed
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    
    return processed_images


def load_image(
    image_file: Union[str, Image.Image],
    input_size: int = 448,
    max_num: int = 12
) -> torch.Tensor:
    """
    Load and preprocess image for model input.
    
    Args:
        image_file: Path to image file or PIL Image
        input_size: Target image size
        max_num: Maximum number of image tiles
        
    Returns:
        Preprocessed image tensor [N, C, H, W]
    """
    if isinstance(image_file, str):
        image = Image.open(image_file).convert('RGB')
    else:
        image = image_file.convert('RGB')
    
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    
    return pixel_values


# ============================================================================
# Data Format Conversion Functions
# ============================================================================

def convert_to_grpo_format(
    input_file: str,
    output_file: str,
    add_assistant_message: bool = True
) -> None:
    """
    Convert dataset to GRPO format (add assistant message if needed).
    
    This function converts a JSONL file with 'solution' field to include
    the assistant message in the messages list.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        add_assistant_message: Whether to add assistant message from solution
    """
    print(f"Converting {input_file} to {output_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            data = json.loads(line)
            messages = data.get('messages', [])
            solution = data.get('solution')
            
            if solution and add_assistant_message:
                # Check if assistant message already exists
                has_assistant = any(m.get('role') == 'assistant' for m in messages)
                if not has_assistant:
                    messages.append({
                        "role": "assistant",
                        "content": solution
                    })
                    data['messages'] = messages
            
            # Write the modified data
            fout.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print("✅ Conversion complete.")


def validate_dataset(file_path: str, sample_size: int = 5) -> dict:
    """
    Validate dataset format and return statistics.
    
    Args:
        file_path: Path to JSONL dataset file
        sample_size: Number of samples to display
        
    Returns:
        Dictionary with validation statistics
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    stats = {
        'total_entries': 0,
        'missing_fields': {'messages': 0, 'images': 0, 'solution': 0},
        'samples': []
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            stats['total_entries'] += 1
            
            # Check required fields
            if 'messages' not in data:
                stats['missing_fields']['messages'] += 1
            if 'images' not in data:
                stats['missing_fields']['images'] += 1
            if 'solution' not in data:
                stats['missing_fields']['solution'] += 1
            
            # Collect samples
            if idx < sample_size:
                stats['samples'].append(data)
    
    return stats


def print_dataset_stats(file_path: str, sample_size: int = 3) -> None:
    """
    Print dataset statistics and samples.
    
    Args:
        file_path: Path to JSONL dataset file
        sample_size: Number of samples to display
    """
    stats = validate_dataset(file_path, sample_size)
    
    print(f"\n{'='*60}")
    print(f"Dataset: {os.path.basename(file_path)}")
    print(f"{'='*60}")
    print(f"Total entries: {stats['total_entries']}")
    print(f"\nMissing fields:")
    for field, count in stats['missing_fields'].items():
        if count > 0:
            print(f"  - {field}: {count} ({count/stats['total_entries']*100:.1f}%)")
    
    print(f"\n{'='*60}")
    print(f"Sample entries ({sample_size}):")
    print(f"{'='*60}")
    for idx, sample in enumerate(stats['samples'], 1):
        print(f"\nSample {idx}:")
        print(json.dumps(sample, ensure_ascii=False, indent=2))
