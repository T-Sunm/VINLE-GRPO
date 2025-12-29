"""
Rewards module for VINLE-GRPO.

Pure scorer logic without ORM dependencies.
ORM wrappers are created in ms-swift plugin file.
"""

from .base_rewards import BaseRewardScorer
from .format_reward import validate_format_tags, TAG_CONFIGS
from .accuracy_reward import AccuracyRewardScorer, get_accuracy_scorer
from .explanation_reward import ExplanationRewardScorer, get_explanation_scorer


__all__ = [
    'BaseRewardScorer',
    'validate_format_tags',
    'TAG_CONFIGS',
    'AccuracyRewardScorer',
    'get_accuracy_scorer',
    'ExplanationRewardScorer',
    'get_explanation_scorer',
]
