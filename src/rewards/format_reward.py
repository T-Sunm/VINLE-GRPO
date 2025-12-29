"""
Format validation utilities for VINLE-GRPO.

Pure logic functions without ORM dependency.
"""

import re
from typing import List


def validate_format_tags(
    completions: List[str],
    required_tags: List[str]
) -> List[float]:
    """
    Validate that completions contain required tags.
    
    Args:
        completions: List of model outputs
        required_tags: List of required tag names (e.g., ["REASONING", "CONCLUSION"])
        
    Returns:
        List of scores (0.0 - 1.0)
    """
    num_tags = len(required_tags)
    base_weight = 1.0 / num_tags if num_tags > 0 else 0.0
    penalty_factor = (base_weight / num_tags * 2) if num_tags > 0 else 0.0
    
    scores = []
    
    for content in completions:
        if not content or not content.strip():
            scores.append(0.0)
            continue
        
        b_total = 0.0  # Bonus
        p_total = 0.0  # Penalty
        
        for tag in required_tags:
            n_open = len(re.findall(fr"<{tag}>", content))
            n_close = len(re.findall(fr"</{tag}>", content))
            n_pair = len(re.findall(fr"<{tag}>.*?</{tag}>", content, re.DOTALL))
            
            # Bonus
            if n_pair >= 1:
                b_tag = base_weight
            elif n_open > 0 or n_close > 0:
                b_tag = base_weight * 0.5
            else:
                b_tag = 0.0
            
            b_total += b_tag
            
            # Penalty for excess tags
            excess_count = max(0, n_open + n_close - 2)
            p_total += excess_count * penalty_factor
        
        total = max(0.0, min(1.0, b_total - p_total))
        scores.append(total)
    
    return scores


# Preset tag configurations
TAG_CONFIGS = {
    'full_grpo': ["REASONING", "CONCLUSION", "EXPLANATION"],
    'think_answer': ["REASONING", "CONCLUSION"],
    'explain_answer': ["CONCLUSION", "EXPLANATION"],
}
