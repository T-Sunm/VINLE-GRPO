import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from .base_model import VQAModel
from .utils import get_system_prompt, parse_output

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MiniCPMModel(VQAModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_path = "/mnt/dataset1/pretrained_fm/openbmb_MiniCPM-o-2-6"
        self._set_clean_model_name()
        self.load_model()

    def load_model(self):
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16,
            init_vision=True,
            init_audio=False,
            init_tts=False,
            low_cpu_mem_usage=True,
            device_map=DEVICE
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True)
        self.model.init_tts()

    def infer(self, question: str, image_path: str) -> tuple[str, str]:
        system_instruction = get_system_prompt()

        user_content = f"Câu hỏi: {question}"
        image = Image.open(image_path).convert('RGB')
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": [
                image, user_content                
            ]}
        ]

        with torch.no_grad():
            res = self.model.chat(
                image=image,
                msgs=messages,
                tokenizer=self.tokenizer
            )

        return parse_output(res)