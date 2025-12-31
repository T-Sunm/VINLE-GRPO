import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple, Union
import re

try:
    from transformers import Qwen2VLForConditionalGeneration
    QWEN_VL_AVAILABLE = True
except ImportError:
    QWEN_VL_AVAILABLE = False
    print("Warning: Qwen2VLForConditionalGeneration not available. Install transformers>=4.37.0")

DEFAULT_SYSTEM_PROMPT = """You are an intelligent assistant that converts short answers into complete sentences. 
Given a question and a short answer, generate a single sentence that incorporates the answer. 
Use the exact words from the provided answer. Do not include the question in your response. 
Return only the sentence, nothing else."""

DEFAULT_USER_PROMPT_TEMPLATE = """Question: {question}
Answer: {answer}

Generate a single sentence answer:"""

# Vietnamese 
VI_SYSTEM_PROMPT = """Bạn là một trợ lý thông minh có nhiệm vụ chuyển đổi các câu trả lời ngắn thành các câu hoàn chỉnh. 
Khi nhận được một câu hỏi và một câu trả lời ngắn, hãy tạo ra một câu đơn duy nhất có chứa nội dung câu trả lời đó. 
Hãy sử dụng chính xác các từ ngữ trong câu trả lời đã cho. Không được đưa câu hỏi vào trong phản hồi của bạn. 
Chỉ trả về duy nhất câu kết quả, không thêm bất kỳ nội dung nào khác."""

VI_USER_PROMPT_TEMPLATE = """Câu hỏi: {question}
Câu trả lời: {answer}

Hãy tạo ra một câu trả lời dạng câu đơn:"""


def load_vintern_model(model_path: str, device: Optional[torch.device] = None):
    """
    Load the Vintern model and tokenizer for synthetic answer generation.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f'Loading Vintern model from: {model_path}')
    print(f'Using device: {device}')
    
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        use_fast=False
    )
    
    print('Vintern model loaded successfully!')
    return model, tokenizer, device


def load_qwen_text_model(
    model_path: str = "/mnt/dataset1/pretrained_fm/Qwen_Qwen2.5-3B-Instruct",
    device: Optional[torch.device] = None
) -> Tuple:
    """
    Load a standard Qwen 2.5/3 text-only model (NOT vision-language).
    """
    import os
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f'Loading Qwen text model from: {model_path}')
    print(f'Using device: {device}')
    
    is_local = os.path.exists(model_path)
    
    # Use device index from CUDA_VISIBLE_DEVICES if set
    device_str = str(device)
    if 'cuda' in device_str:
        # Just use the device string directly
        device_map = device_str
    else:
        device_map = "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=device_map,  # Use specific device instead of auto
        local_files_only=is_local
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=is_local
    )
    
    print(f'Qwen text model loaded successfully on {device_map}!')
    return model, tokenizer, device


def generate_synthetic_answer_qwen_text(
    model, 
    tokenizer, 
    question: str, 
    answer: str, 
    device: torch.device, 
    max_new_tokens: int = 128,
    system_prompt: str = None,
    user_prompt_template: str = None
) -> str:
    """
    Generate a synthetic answer using Qwen text model.
    
    Args:
        model: The loaded Qwen model
        tokenizer: The loaded tokenizer
        question: The question string
        answer: The short answer string
        device: The device to use for inference
        max_new_tokens: Maximum number of new tokens to generate
        system_prompt: Custom system prompt (uses DEFAULT_SYSTEM_PROMPT if None)
        user_prompt_template: Custom user template with {question} and {answer} placeholders
        
    Returns:
        Generated synthetic answer as a full sentence
    """
    # Use default prompts if not provided
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    if user_prompt_template is None:
        user_prompt_template = DEFAULT_USER_PROMPT_TEMPLATE
    
    # Format user content
    user_content = user_prompt_template.format(question=question, answer=answer.strip())
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    # Clean up thinking tags if present
    if "<think>" in generated_text:
        generated_text = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL).strip()
    
    if not generated_text:
        generated_text = answer
    
    return generated_text

def generate_synthetic_answers_batch_qwen(model, processor, data_list: List[Dict], device: torch.device, question_key: str = 'question', answer_key: str = 'ground_truth', max_new_tokens: int = 128) -> List[Dict]:
    """
    Generate synthetic answers for a list of question-answer pairs.
    
    Args:
        model: The loaded Vintern model
        tokenizer: The loaded tokenizer  
        data_list: List of dicts with question and answer keys
        device: The device to use for inference
        question_key: Key for question in data dict (default: 'question')
        answer_key: Key for answer in data dict (default: 'ground_truth')
        
    Returns:
        List of dicts with added 'syn_ans' key containing the synthetic answer
    """
    results = []
    
    for item in tqdm(data_list, desc='Generating synthetic answers'):
        question = item[question_key]
        answer = item[answer_key]
        
        try:
            syn_ans = generate_synthetic_answer_qwen_text(model, processor, question, answer, device, max_new_tokens)
        except Exception as e:
            print(f'Error generating synthetic answer for question: {question[:50]}... Error: {e}')
            syn_ans = answer  
        
        result = item.copy()
        result['syn_ans'] = syn_ans
        results.append(result)
    
    return results


def create_syn_ans_prompt(question: str, answer: str, system_prompt: str = None, user_prompt_template: str = None) -> str:
    """
    Create a prompt for synthetic answer generation.
    
    Args:
        question: The question string
        answer: The short answer string
        system_prompt: Custom system prompt (uses DEFAULT_SYSTEM_PROMPT if None)
        user_prompt_template: Custom user template with {question} and {answer} placeholders
        
    Returns:
        Formatted prompt string for the model
    """
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    if user_prompt_template is None:
        user_prompt_template = DEFAULT_USER_PROMPT_TEMPLATE
    
    user_content = user_prompt_template.format(question=question, answer=answer.strip())
    prompt = f"{system_prompt}\n\n{user_content}"
    return prompt


def generate_synthetic_answer_vintern(
    model, 
    tokenizer, 
    question: str, 
    answer: str, 
    device: torch.device, 
    max_new_tokens: int = 128,
    system_prompt: str = None,
    user_prompt_template: str = None
) -> str:
    """
    Generate a synthetic answer using Vintern model's .chat() method.
    
    Args:
        model: The loaded Vintern model
        tokenizer: The loaded tokenizer
        question: The question string
        answer: The short answer string
        device: The device to use for inference
        max_new_tokens: Maximum number of new tokens to generate
        system_prompt: Custom system prompt (uses DEFAULT_SYSTEM_PROMPT if None)
        user_prompt_template: Custom user template with {question} and {answer} placeholders
        
    Returns:
        Generated synthetic answer as a full sentence
    """
    prompt = create_syn_ans_prompt(question, answer, system_prompt, user_prompt_template)
    
    with torch.no_grad():
        response = model.chat(
            tokenizer,
            None,  # pixel_values = None for text-only inference
            prompt,
            generation_config={
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
        )
    
    generated_text = response.strip() if response else answer
    
    return generated_text


def generate_synthetic_answers_batch_vintern(model, tokenizer, data_list: List[Dict], device: torch.device, question_key: str = 'question', answer_key: str = 'ground_truth', max_new_tokens: int = 128, system_prompt = None, user_prompt_template = None) -> List[Dict]:
    """
    Generate synthetic answers for a list of question-answer pairs.
    
    Args:
        model: The loaded Vintern model
        tokenizer: The loaded tokenizer  
        data_list: List of dicts with question and answer keys
        device: The device to use for inference
        question_key: Key for question in data dict (default: 'question')
        answer_key: Key for answer in data dict (default: 'ground_truth')
        
    Returns:
        List of dicts with added 'syn_ans' key containing the synthetic answer
    """
    results = []
    
    for item in tqdm(data_list, desc='Generating synthetic answers'):
        question = item[question_key]
        answer = item[answer_key]
        
        try:
            syn_ans = generate_synthetic_answer_vintern(model, tokenizer, question, answer, device, max_new_tokens, system_prompt, user_prompt_template)
        except Exception as e:
            print(f'Error generating synthetic answer for question: {question[:50]}... Error: {e}')
            syn_ans = answer  
        
        result = item.copy()
        result['syn_ans'] = syn_ans
        results.append(result)
    
    return results

