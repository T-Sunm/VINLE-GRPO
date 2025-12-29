"""
Shared tag parsers for VINLE-GRPO inference.
Parses uppercase tags: REASONING, CONCLUSION, EXPLANATION
"""

import re


def extract_tag(text: str, tag: str) -> str:
    """
    Extract content from XML-like tags (case-insensitive).
    
    Args:
        text: Model response
        tag: Tag name (REASONING, CONCLUSION, or EXPLANATION)
    
    Returns:
        Extracted content or empty string
    """
    match = re.search(fr'<{tag}>(.*?)</{tag}>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Fallback: find opening tag
    match_open = re.search(fr'<{tag}>(.*?)(?=<|$)', text, re.DOTALL | re.IGNORECASE)
    if match_open:
        return match_open.group(1).strip()
    
    return ""


def parse_grpo_output(response: str) -> dict:
    """
    Parse GRPO output with 3 tags: REASONING + CONCLUSION + EXPLANATION.
    
    Args:
        response: Model output
    
    Returns:
        dict with keys: reasoning, answer, explanation, raw_response
    """
    return {
        "reasoning": extract_tag(response, "REASONING"),
        "answer": extract_tag(response, "CONCLUSION"),
        "explanation": extract_tag(response, "EXPLANATION"),
        "raw_response": response
    }


def parse_ota_output(response: str) -> dict:
    """
    Parse OTA output with 2 tags: REASONING + CONCLUSION.
    
    Returns:
        dict with keys: reasoning, answer, explanation (empty), raw_response
    """
    return {
        "reasoning": extract_tag(response, "REASONING"),
        "answer": extract_tag(response, "CONCLUSION"),
        "explanation": "",  # No explanation in OTA
        "raw_response": response
    }


def parse_oea_output(response: str) -> dict:
    """
    Parse OEA output with 2 tags: CONCLUSION + EXPLANATION.
    
    Returns:
        dict with keys: reasoning (empty), answer, explanation, raw_response
    """
    return {
        "reasoning": "",  # No reasoning in OEA
        "answer": extract_tag(response, "CONCLUSION"),
        "explanation": extract_tag(response, "EXPLANATION"),
        "raw_response": response
    }


def parse_sft_output(response: str) -> dict:
    """
    Parse SFT output with 2 tags: CONCLUSION + EXPLANATION.
    
    Returns:
        dict with keys: reasoning (empty), answer, explanation, raw_response
    """
    return {
        "reasoning": "",  # No reasoning in SFT
        "answer": extract_tag(response, "CONCLUSION"),
        "explanation": extract_tag(response, "EXPLANATION"),
        "raw_response": response
    }
