"""
Common utilities for VINLE-GRPO inference.
"""

from .prompts import get_grpo_prompt, get_ota_prompt, get_oea_prompt, get_sft_prompt
from .parsers import (
    extract_tag,
    parse_grpo_output,
    parse_ota_output, 
    parse_oea_output,
    parse_sft_output
)
from .processors import normalize_data_item, process_inference_sample

__all__ = [
    # Prompts
    "get_grpo_prompt",
    "get_ota_prompt",
    "get_oea_prompt",
    "get_sft_prompt",
    # Parsers
    "extract_tag",
    "parse_grpo_output",
    "parse_ota_output",
    "parse_oea_output",
    "parse_sft_output",
    # Processors
    "normalize_data_item",
    "process_inference_sample",
]
