import torch
from transformers import AutoModelForCausalLM
from PIL import Image

from .base_model import VQAModel
# from .utils import get_system_prompt, parse_output
from .utils import get_system_prompt, parse_output, get_grpo_system_prompt, parse_output_grpo

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OvisModel(VQAModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_path = "/mnt/dataset1/pretrained_fm/AIDC-AI_Ovis2.5-9B"
        self._set_clean_model_name()
        self.load_model()

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16, multimodal_max_length=32768,
            trust_remote_code=True).eval().to(DEVICE)

    def infer(self, question: str, image_path: str) -> tuple[str, str]:
        system_instruction = get_system_prompt()

        user_content = f"Câu hỏi: {question}"
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user",
            "content": [
                {"type": "image", "image": Image.open(image_path).convert('RGB')},
                {"type": "text", "text": user_content},
            ],
        }]
        
        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(messages, add_generation_prompt=True,
                                                                          enable_thinking=False)
        input_ids = input_ids.cuda()
        pixel_values = pixel_values.cuda() if pixel_values is not None else None
        grid_thws = grid_thws.cuda() if grid_thws is not None else None

        with torch.inference_mode():
            outputs = self.model.generate(
                inputs=input_ids,
                pixel_values=pixel_values,
                grid_thws=grid_thws,
                enable_thinking=False,
                enable_thinking_budget=False,
                max_new_tokens=100,
                thinking_budget=0,
                eos_token_id=self.model.text_tokenizer.eos_token_id
            )

        response = self.model.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return parse_output(response) 

    def infer_grpo(self, question: str, image_path: str) -> tuple[str, str]:
        system_instruction = get_grpo_system_prompt(question)

        user_content = f"""Now, answer this question based on the image: 
        Question: {question}. 
        Let's response in three tag pairs in your response: <think></think>, <answer></answer>, <explain></explain>."""
        messages = [
        {"role": "user",
        "content": [
            {"type": "image", "image": Image.open(image_path).convert("RGB")},
            {"type": "text",
            "text": (
                "Now answer using exactly three tag pairs: "
                "<think></think>, <answer></answer>, <explain></explain>.\n"
                f"Question: {question}"
            )},
        ]}
        ]

        
        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(messages, add_generation_prompt=True,
                                                                          enable_thinking=False)
        input_ids = input_ids.cuda()
        pixel_values = pixel_values.cuda() if pixel_values is not None else None
        grid_thws = grid_thws.cuda() if grid_thws is not None else None

        with torch.inference_mode():
            outputs = self.model.generate(
                inputs=input_ids,
                pixel_values=pixel_values,
                grid_thws=grid_thws,
                enable_thinking=False,
                enable_thinking_budget=False,
                max_new_tokens=100,
                thinking_budget=0,
                eos_token_id=self.model.text_tokenizer.eos_token_id
            )

        response = self.model.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return parse_output_grpo(response) 