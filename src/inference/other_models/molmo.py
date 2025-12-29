import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers import GenerationConfig
import numpy as np

from .base_model import VQAModel
from .utils import get_system_prompt, parse_output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MoLMo requires this patch
orig_stack = np.stack
def patched_stack(arrays, axis=0, *args, **kwargs):
    if "dtype" in kwargs:
        kwargs.pop("dtype")
    return orig_stack(arrays, axis=axis, *args, **kwargs)
np.stack = patched_stack


class MolmoModel(VQAModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_path = "/mnt/dataset1/pretrained_fm/allenai_Molmo-7B-D-0924"
        self._set_clean_model_name()
        self.load_model()

    def load_model(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map='auto',
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).eval().cuda()
        self.tokenizer = self.processor.tokenizer

    def infer(self, question: str, image_path: str) -> tuple[str, str]:
        system_instruction = get_system_prompt()

        user_content = f"Câu hỏi: {question}"
        prompt = system_instruction + user_content

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor.process(images=[image], text=prompt)
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
        inputs["images"] = inputs["images"].to(torch.bfloat16)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
                output = self.model.generate_from_batch(
                    inputs,
                    GenerationConfig(max_new_tokens=100, stop_strings="<|endoftext|>"),
                    tokenizer=self.tokenizer
                )

        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True)

        return parse_output(generated_text) 