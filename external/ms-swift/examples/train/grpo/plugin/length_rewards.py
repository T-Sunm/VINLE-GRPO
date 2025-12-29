def length_penalty_answer(pred, truth, ratio=1.3):
    pred_len = len(pred.split())
    truth_len = len(truth.split())
    max_ok = round(truth_len * ratio) # mở rộng biên độ chấp nhận
    
    if pred_len <= max_ok:
        return 1.0
    else:
        excess_ratio = (pred_len - max_ok) / truth_len
        return - min(excess_ratio, 1.0)


def length_penalty_explanation(pred, truth, ratio=1.2, sentence_penalty_weight=0.3):
    """
    Penalty function với kiểm soát số câu:
    - Nếu pred có >= 2 câu (phân tách bởi '.' hoặc ',') -> phạt dựa trên số câu
    - Nếu pred chỉ 1 câu -> kiểm tra độ dài như bình thường
    
    Args:
        pred: Câu prediction
        truth: Câu ground truth
        ratio: Tỷ lệ độ dài tối đa cho phép
        sentence_penalty_weight: Trọng số phạt cho mỗi câu thừa (mặc định 0.3)
    
    Returns:
        float: 1.0 (ok), -0.5 (quá ngắn), hoặc giá trị âm (quá dài/nhiều câu)
    """
    # Tách câu dựa trên dấu '.' (loại bỏ câu rỗng)
    import re
    pred_sentences = [s.strip() for s in re.split(r'[.]', pred) if s.strip()]
    truth_sentences = [s.strip() for s in re.split(r'[.]', truth) if s.strip()]
    
    num_pred_sentences = len(pred_sentences)
    num_truth_sentences = len(truth_sentences)
    
    # Kiểm tra số câu: Nếu pred có >= 2 câu mà truth chỉ 1 câu -> phạt
    if num_truth_sentences == 1 and num_pred_sentences >= 2:
        # Phạt càng nặng khi gen càng nhiều câu
        # Ví dụ: 2 câu -> -0.3, 3 câu -> -0.6, 4 câu -> -0.9, 5+ câu -> -1.0
        excess_sentences = num_pred_sentences - 1
        penalty = -min(excess_sentences * sentence_penalty_weight, 1.0)
        return penalty
    
    # Nếu số câu hợp lệ -> kiểm tra độ dài từ
    pred_len = len(pred.split())
    truth_len = len(truth.split())
    min_ok = max(5, int(truth_len * 0.7))
    max_ok = round(truth_len * ratio)
    
    if min_ok <= pred_len <= max_ok:
        return 1.0
    elif pred_len < min_ok:
        return -0.5
    else:
        excess_ratio = (pred_len - max_ok) / truth_len
        return -min(excess_ratio, 1.0)



# def standard_len(pred, truth):
#     pred_len = len(pred.split(``))
#     mean = pred_len / len(pred)
#     std = 


# def length_reasoning(pred, truth, alpha = 0):
#     pred_len = len(pred.split())
#     truth_len = len(truth.split())

# import numpy as np
# import math

# def calculate_standardized_length(current_length: int, all_correct_lengths: list[int], epsilon: float = 1e-6) -> float:
    
#     if not all_correct_lengths:
#         return 0.0

#     mu = np.mean(all_correct_lengths)
#     sigma = np.std(all_correct_lengths)
    
#     # z = (|o| - μ) / (σ + ε)
#     z = (current_length - mu) / (sigma + epsilon)
    
#     return z

# def length_reasoning(z_score: float, is_correct: bool, alpha: float = 0.05) -> float:
#     """
#     Args:
#         z_score: From calculate_standardized_length).
#         is_correct: Boolean, True if o_i true, otherwise False 
#         alpha

#     Returns:
#         float: reward
#     """
    
#     if is_correct:
#         # R = exp(-α * z) 
#         return math.exp(-alpha * z_score)
#     else:
#         #R = -1 
#         return -1.0

# class LengthPenaltyExplanationReward(ORM):
#     """
#     Length penalty reward cho explanation.
#     Kiểm tra độ dài explanation có nằm trong khoảng cho phép không.
#     """
#     def __init__(self, alpha=1):
#         self.alpha = alpha
    
#     def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        
#         return rewards