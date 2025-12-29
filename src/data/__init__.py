"""
Data processing module for VINLE-GRPO.

This module provides functionality for:
- Loading and converting ViVQA-X dataset to GRPO/SFT format
- Image preprocessing and transformation
- Dataset validation and statistics

Main entry points:
- load_vivqa_for_grpo(): Load and convert ViVQA-X to GRPO format
- ViVQAProcessor: Process and analyze ViVQA-X dataset
"""

from .dataset_loader import (
    load_vivqa_for_grpo,
    SYSTEM_PROMPTS,
    USER_CONTENT_TEMPLATES
)

from .vivqa_processor import (
    ViVQAProcessor,
    ViVQAXSample
)

from .data_utils import (
    build_transform,
    load_image,
    dynamic_preprocess,
    convert_to_grpo_format,
    validate_dataset,
    print_dataset_stats
)


__all__ = [
    # Main loaders
    'load_vivqa_for_grpo',
    'SYSTEM_PROMPTS',
    'USER_CONTENT_TEMPLATES',
    
    # ViVQA processing
    'ViVQAProcessor',
    'ViVQAXSample',
    
    # Utilities
    'build_transform',
    'load_image',
    'dynamic_preprocess',
    'convert_to_grpo_format',
    'validate_dataset',
    'print_dataset_stats',
]
