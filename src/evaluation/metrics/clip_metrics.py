import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from ..core.shared_models import SharedCLIPScoreModel
from ..core.text_preprocessing import preprocess_for_nlg_metrics

def _to_clip_image_tensor(image_paths: list[str]) -> torch.Tensor:
    """
    Load images from paths and convert to a single tensor [N, 3, 224, 224] uint8.
    Note: Using 224x224 as it's the default for most CLIP models.
    """
    tensors = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((224, 224))
            tensors.append(torch.from_numpy(np.array(img)).permute(2, 0, 1))
        except Exception as e:
            print(f"Warning: Failed to load image {path}: {e}")
            # Fallback to black image if one fails to keep batch alignment
            tensors.append(torch.zeros((3, 224, 224), dtype=torch.uint8))
            
    return torch.stack(tensors)

@torch.inference_mode()
def compute_clip_scores(
    image_paths: list[str],
    texts: list[str],
    device: str = "cuda",
    model_name_or_path: str = "openai/clip-vit-base-patch16",
    batch_size: int = 8,  # Reduced batch size for CLIP safety
    max_len: int = 150,
) -> list[float]:
    """
    Compute CLIPScore(image, text) for each sample.
    
    Args:
        image_paths: List of local paths to images
        texts: List of strings (explanations/answers)
        device: Computation device
        model_name_or_path: CLIP model identifier
        batch_size: Batch size for computation
        max_len: Maximum text length for preprocessing
        
    Returns:
        List[float] scores (scaled 0-100), length == len(texts)
    """
    if not texts or not image_paths:
        return []

    if len(image_paths) != len(texts):
        raise ValueError(f"Number of images ({len(image_paths)}) != number of texts ({len(texts)})")

    # Preprocess texts
    processed_texts = [preprocess_for_nlg_metrics(t, max_len=max_len) for t in texts]

    # Get singleton model
    metric = SharedCLIPScoreModel.get_instance(model_name_or_path=model_name_or_path, device=device)

    scores = [0.0] * len(texts)

    # Process in batches
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_paths = image_paths[start:end]
        batch_texts = processed_texts[start:end]

        try:
            batch_imgs = _to_clip_image_tensor(batch_paths).to(device)
            
            # CLIPScore in torchmetrics can take batch of images and texts
            # Returns a scalar tensor which is the mean by default.
            # To get individual scores, we can call it per sample or check if it supports batch return.
            # Most efficient is usually batch processing, but torchmetrics CLIPScore returns mean of batch.
            # We'll compute individual scores to be safe for per-sample logging.
            
            for i, (img, txt) in enumerate(zip(batch_imgs, batch_texts), start=start):
                # img is [3, 224, 224], CLIPScore expects [1, 3, 224, 224] for single sample
                s = metric(img.unsqueeze(0), [txt])
                scores[i] = float(s.detach().cpu().item())
                
        except Exception as e:
            print(f"Warning: CLIPScore batch failed at {start}-{end}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return scores
