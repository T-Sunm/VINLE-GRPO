import torch
import numpy as np
from PIL import Image
from ..core.text_preprocessing import preprocess_for_nlg_metrics


class _CLIPScoreComputer:
    """Singleton CLIP model for computing CLIPScore directly via transformers."""
    _instances = {}

    @classmethod
    def get_instance(cls, model_name_or_path: str = "openai/clip-vit-base-patch16",
                     device: str = "cuda"):
        from transformers import CLIPModel, CLIPProcessor

        key = (model_name_or_path, device)
        if key in cls._instances:
            return cls._instances[key]

        model = CLIPModel.from_pretrained(model_name_or_path).to(device)
        model.eval()
        processor = CLIPProcessor.from_pretrained(model_name_or_path)

        cls._instances[key] = (model, processor)
        return model, processor


@torch.inference_mode()
def compute_clip_scores(
    image_paths: list[str],
    texts: list[str],
    device: str = "cuda",
    model_name_or_path: str = "openai/clip-vit-base-patch16",
    batch_size: int = 8,
    max_len: int = 150,
) -> list[float]:
    """
    Compute CLIPScore(image, text) for each sample.
    
    CLIPScore = max(100 * cos_sim(image_emb, text_emb), 0)
    
    Returns:
        List[float] scores (scaled 0-100), length == len(texts)
    """
    if not texts or not image_paths:
        return []

    if len(image_paths) != len(texts):
        raise ValueError(f"Number of images ({len(image_paths)}) != number of texts ({len(texts)})")

    # Preprocess texts
    processed_texts = [preprocess_for_nlg_metrics(t, max_len=max_len) for t in texts]

    # Get model
    model, processor = _CLIPScoreComputer.get_instance(
        model_name_or_path=model_name_or_path, device=device
    )

    scores = [0.0] * len(texts)

    gpu_failed = False

    for start in range(0, len(texts), batch_size):
        if gpu_failed:
            break

        end = min(start + batch_size, len(texts))
        batch_paths = image_paths[start:end]
        batch_texts = processed_texts[start:end]

        # Sanitize texts: replace empty/whitespace with a placeholder
        batch_texts = [t if t and t.strip() else "image" for t in batch_texts]

        try:
            # Load images
            images = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(img)
                except Exception:
                    images.append(Image.new("RGB", (224, 224), (0, 0, 0)))

            # Process inputs on CPU first
            inputs = processor(
                text=batch_texts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get embeddings
            outputs = model(**inputs)
            image_embeds = outputs.image_embeds  # [B, D]
            text_embeds = outputs.text_embeds     # [B, D]

            # Normalize
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            # Cosine similarity per pair, scaled to 0-100
            cos_sim = (image_embeds * text_embeds).sum(dim=-1)  # [B]
            clip_scores = torch.clamp(cos_sim * 100, min=0).cpu().tolist()

            for i, score in enumerate(clip_scores):
                scores[start + i] = score

        except (torch.cuda.CudaError, RuntimeError) as e:
            err_str = str(e).lower()
            if "cuda" in err_str or "device-side" in err_str:
                print(f"Warning: CLIPScore GPU error at batch {start}-{end}, stopping CLIPScore computation.")
                gpu_failed = True
            else:
                print(f"Warning: CLIPScore batch failed at {start}-{end}: {e}")
        except Exception as e:
            print(f"Warning: CLIPScore batch failed at {start}-{end}: {e}")

    if gpu_failed:
        print("CLIPScore: GPU crashed, scores may be incomplete.")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return scores

