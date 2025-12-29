"""
Shared data processing utilities for inference.
"""

from pathlib import Path


def normalize_data_item(item: dict) -> dict:
    """
    Normalize dataset item with standard fields.
    
    Ensures:
    - image_name exists (from image_id if needed)
    - answer exists (from answers list if needed)
    
    Args:
        item: Dataset item
    
    Returns:
        Normalized item (modified in-place)
    """
    # Generate image_name from image_id if missing
    if "image_name" not in item and "image_id" in item:
        item["image_name"] = f"COCO_val2014_{int(item['image_id']):012d}.jpg"
    
    # Extract answer from answers list if needed
    if "answer" not in item and "answers" in item:
        item["answer"] = item["answers"][0]["answer"]
    
    return item


def process_inference_sample(model, item: dict, image_folder: Path, infer_fn) -> dict:
    """
    Generic sample processor for all inference modes.
    
    Handles:
    - Data normalization
    - Image path validation
    - Inference execution with error handling
    - Result update
    
    Args:
        model: Model instance (VinternModel or other)
        item: Dataset item
        image_folder: Path to images directory
        infer_fn: Inference function with signature (model, question, image_path) -> dict
    
    Returns:
        Updated item with predictions
    """
    # Normalize fields
    normalize_data_item(item)
    
    # Build and validate image path
    img_path = image_folder / item["image_name"]
    
    if not img_path.exists():
        # Image not found - set error
        item.update({
            "pred_reasoning": "",
            "pred_answer": "ERROR: Image not found",
            "pred_explanation": "",
            "raw_response": ""
        })
        return item
    
    # Run inference with error handling
    try:
        result = infer_fn(model, item["question"], str(img_path))
        
        # Map to requested format
        update_dict = {
            "thinking": result.get("reasoning", ""),
            "predict": result.get("answer", ""),
            "pred_explanation": result.get("explanation", ""),
            # Keep raw response for debugging/logging, or optional
            # "raw_response": result.get("raw_response", "") 
        }
        item.update(update_dict)
        
    except Exception as e:
        # Inference failed - set error
        item.update({
            "thinking": "",
            "predict": f"ERROR: {e}",
            "pred_explanation": ""
        })
    
    return item
