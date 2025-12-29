"""
Shared prompts for VINLE-GRPO inference.
All prompts use uppercase tags: REASONING, CONCLUSION, EXPLANATION
"""


def get_grpo_prompt(question: str) -> str:
    """
    GRPO prompt with 3 tags: REASONING + CONCLUSION + EXPLANATION.
    Used for: Full GRPO training and ALL zero-shot models.
    """
    return f"""<image> Bạn là một trợ lý ngôn ngữ thị giác hữu ích, được thiết kế cho suy luận có cấu trúc.
Khi trả lời các câu hỏi về hình ảnh, bạn phải trả lời chính xác trong ba giai đoạn, mỗi giai đoạn bắt buộc phải tuân theo format:
<REASONING>[Đưa ra phân tích lập luận chi tiết, từng bước để giải quyết vấn đề.]</REASONING>
<CONCLUSION>[Nêu câu trả lời cuối cùng là một từ hoặc cụm từ.]</CONCLUSION>
<EXPLANATION>[Tổng hợp các thông tin từ REASONING và cho ra câu mô tả ngắn gọn các phân tích đặc điểm.] Hình ảnh cho thấy...</EXPLANATION>

Vui lòng áp dụng định dạng này một cách tỉ mỉ để phân tích hình ảnh được cung cấp và trả lời câu hỏi: {question}
Câu trả lời:""".strip()


def get_ota_prompt(question: str) -> str:
    """
    OTA ablation prompt with 2 tags: REASONING + CONCLUSION.
    Used for: GRPO training without explanation reward.
    """
    return f"""<image> Bạn là một trợ lý ngôn ngữ thị giác hữu ích, được thiết kế cho suy luận có cấu trúc.
Khi trả lời các câu hỏi về hình ảnh, bạn phải trả lời chính xác trong hai giai đoạn, mỗi giai đoạn bắt buộc phải tuân theo format:
<REASONING>[Đưa ra phân tích lập luận chi tiết, từng bước để giải quyết vấn đề.]</REASONING>
<CONCLUSION>[Nêu câu trả lời cuối cùng là một từ hoặc cụm từ.]</CONCLUSION>

Vui lòng áp dụng định dạng này một cách tỉ mỉ để phân tích hình ảnh được cung cấp và trả lời câu hỏi: {question}
Câu trả lời:""".strip()


def get_oea_prompt(question: str) -> str:
    """
    OEA ablation prompt with 2 tags: CONCLUSION + EXPLANATION.
    Used for: GRPO training without reasoning reward.
    """
    return f"""<image> Bạn là một trợ lý ngôn ngữ thị giác hữu ích, được thiết kế cho suy luận có cấu trúc.
Khi trả lời các câu hỏi về hình ảnh, bạn phải trả lời chính xác trong hai giai đoạn, mỗi giai đoạn bắt buộc phải tuân theo format:
<CONCLUSION>[Nêu câu trả lời cuối cùng là một từ hoặc cụm từ.]</CONCLUSION>
<EXPLANATION>[Giải thích một câu ngắn gọn chứng minh câu trả lời.] Hình ảnh cho thấy...</EXPLANATION>

Vui lòng áp dụng định dạng này một cách tỉ mỉ để phân tích hình ảnh được cung cấp và trả lời câu hỏi: {question}
Câu trả lời:""".strip()


def get_sft_prompt(question: str) -> str:
    """
    SFT baseline prompt with 2 tags: CONCLUSION + EXPLANATION.
    Used for: Supervised fine-tuning baseline.
    """
    return f"""<image> Bạn là một trợ lý ngôn ngữ thị giác hữu ích, được thiết kế cho suy luận có cấu trúc.
Khi trả lời các câu hỏi về hình ảnh, bạn phải trả lời chính xác trong hai giai đoạn, mỗi giai đoạn bắt buộc phải tuân theo format:
<CONCLUSION>[Nêu câu trả lời cuối cùng là một từ hoặc cụm từ.]</CONCLUSION>
<EXPLANATION>[Giải thích một câu ngắn gọn chứng minh câu trả lời.] Hình ảnh cho thấy...</EXPLANATION>

Vui lòng áp dụng định dạng này một cách tỉ mỉ để phân tích hình ảnh được cung cấp và trả lời câu hỏi: {question}
Câu trả lời:""".strip()
