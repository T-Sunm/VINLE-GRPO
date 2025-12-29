import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from .base_model import VQAModel
from .utils import get_system_prompt, parse_output

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VideoLLaMAModel(VQAModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_path = "/mnt/dataset1/pretrained_fm/DAMO-NLP-SG_VideoLLaMA3-7B-Image"
        self._set_clean_model_name()
        self.load_model()

    def load_model(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            device_map=DEVICE
        ).eval()
        self.tokenizer = self.processor.tokenizer

    def infer(self, question: str, image_path: str) -> tuple[str, str]:
        system_instruction = get_system_prompt()
    
        user_content = f"Câu hỏi: {question}"

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": {"image_path": image_path}},
                    {"type": "text", "text": user_content}
                ]
            }
        ]

        inputs = self.processor(conversation=conversation, return_tensors="pt")
        inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor)
                  else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
                output_ids = self.model.generate(**inputs, max_new_tokens=100)

        response = self.processor.batch_decode(
            output_ids, skip_special_tokens=True)[0].strip()

        return parse_output(response) 