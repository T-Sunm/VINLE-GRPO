"""
Basic hard reward functions for ablation study (DeepSeek style).
Reward 1.0 for success, 0.0 for failure.
"""

import re
from typing import List


def basic_format_reward(completions: List[str], **kwargs) -> List[float]:
    """
    Check if <REASONING> and <CONCLUSION> tags are correctly used.
    Reward 1.0 if both exist as pairs, 0.0 otherwise.
    """
    rewards = []
    # Regex for paired tags
    pat_reasoning = re.compile(r"<REASONING>.*?</REASONING>", re.DOTALL | re.IGNORECASE)
    pat_conclusion = re.compile(r"<CONCLUSION>.*?</CONCLUSION>", re.DOTALL | re.IGNORECASE)

    for content in completions:
        if not content:
            rewards.append(0.0)
            continue
            
        # Check presence of both paired structures
        has_reasoning = bool(pat_reasoning.search(content))
        has_conclusion = bool(pat_conclusion.search(content))
        
        if has_reasoning and has_conclusion:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
            
    return rewards


def basic_accuracy_reward(completions: List[str], solution: List[str], **kwargs) -> List[float]:
    """
    Check for exact string match of the answer inside <CONCLUSION> tags.
    Reward 1.0 if match exactly (case-insensitive, stripped), 0.0 otherwise.
    """
    rewards = []
    for content, sol in zip(completions, solution):
        # Extract from CONCLUSION tags in completion
        content_match = re.search(r"<CONCLUSION>(.*?)</CONCLUSION>", content, flags=re.DOTALL | re.IGNORECASE)
        student_answer = content_match.group(1).strip() if content_match else ""
        
        # Extract from CONCLUSION or use full solution string
        sol_match = re.search(r"<CONCLUSION>(.*?)</CONCLUSION>", sol, flags=re.DOTALL | re.IGNORECASE)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
        
        # Exact match logic
        if student_answer and ground_truth and student_answer.lower() == ground_truth.lower():
            rewards.append(1.0)
        else:
            rewards.append(0.0)
            
    return rewards
