"""
Format detector for automatic inference output format detection.

Supports:
- GRPO: has thinking + pred_explanation
- OEA: has pred_explanation only
- OTA: has thinking only
- ZEROSHOT: has thinking + pred_explanation (baseline)
"""

from typing import Dict, List


def detect_format(data: List[Dict]) -> Dict[str, bool]:
    """
    Auto-detect inference output format.
    
    Args:
        data: List of inference result dictionaries
        
    Returns:
        Dictionary with format information:
            - has_thinking: bool
            - has_pred_explanation: bool
            - format_name: str (GRPO, OEA, OTA, ZEROSHOT)
    """
    if not data or len(data) == 0:
        return {
            'has_thinking': False,
            'has_pred_explanation': False,
            'format_name': 'UNKNOWN'
        }
    
    sample = data[0]
    
    has_thinking = bool(sample.get('thinking', '').strip())
    has_pred_explanation = bool(sample.get('pred_explanation', '').strip())
    
    # Determine format name
    if has_thinking and has_pred_explanation:
        format_name = 'GRPO/ZEROSHOT'
    elif has_pred_explanation:
        format_name = 'OEA'
    elif has_thinking:
        format_name = 'OTA'
    else:
        format_name = 'UNKNOWN'
    
    return {
        'has_thinking': has_thinking,
        'has_pred_explanation': has_pred_explanation,
        'format_name': format_name
    }


def validate_format_consistency(data: List[Dict]) -> bool:
    """
    Validate that all samples have consistent format.
    
    Args:
        data: List of inference result dictionaries
        
    Returns:
        True if all samples have the same format, False otherwise
    """
    if not data:
        return True
    
    first_format = detect_format([data[0]])
    
    for item in data[1:]:
        item_format = detect_format([item])
        if (item_format['has_thinking'] != first_format['has_thinking'] or
            item_format['has_pred_explanation'] != first_format['has_pred_explanation']):
            return False
    
    return True


__all__ = [
    'detect_format',
    'validate_format_consistency',
]
