import string
import re
import unicodedata
import torch
from base_rewards import BaseRewardScorer
from pycocoevalcap.rouge.rouge import Rouge
def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    
    # 1. Chuẩn hóa các câu trả lời Có/Không
    if text in ["có", "đúng", "yes", "true", "correct"]:
        return "có"
    if text in ["không", "sai", "no", "false", "incorrect"]:
        return "không"

            
    # 4. Loại bỏ các ký tự đặc biệt và khoảng trắng thừa
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 5. REMOVED - Không sắp xếp từ vì nó phá hủy nghĩa!
    # Ví dụ: "xe đỏ" != "đỏ xe"
    
    return text

def clean_text(text: str) -> str:
    """
    Làm sạch chuỗi trước khi gửi tới pycocoevalcap.Meteor
    - Thay thế mọi biến thể xuống dòng và '|||' bằng khoảng trắng
    - Loại bỏ ký tự điều khiển (Unicode category == 'Cc')
    - Gom nhiều khoảng trắng liên tiếp thành 1
    """
    text = (
        text.replace("|||", " ")
        .replace("\r\n", " ")
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("\u2028", " ")
        .replace("\u2029", " ")
    )
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cc")
    text = re.sub(r"\s+", " ", text).strip()
    return text

class AccuracyRewardScorer(BaseRewardScorer):
    """
    Accuracy reward scorer sử dụng hybrid approach: ROUGE-L + BERTScore (PhoBERT)
    Trả về điểm kết hợp (0.0 - 1.0).
    """
    def __init__(self, alpha: float = 0.5, threshold: float = 0.3):
        """
        Args:
            alpha: Trọng số cho BERTScore (1-alpha cho ROUGE-L)
                  alpha=0.5 => 50% BERTScore, 50% ROUGE-L
            threshold: Ngưỡng tối thiểu, nếu reward < threshold thì gán -1.0
        """
        self.initialize_bertscore(model_name_or_path="phobert")
        self.rouge_scorer = Rouge()
        self.alpha = alpha
        self.threshold = threshold
    
    def calculate_rouge_batch(self, ground_truths: dict, predictions: dict) -> dict:
        """
        Tính ROUGE-L cho batch predictions.
        
        Args:
            ground_truths: {id: ground_truth_string}
            predictions: {id: prediction_string}
            
        Returns:
            {id: rouge_l_score}
        """
        if not predictions:
            return {}
        
        gts = {}
        res = {}
        valid_ids = []
        
        for id_, pred in predictions.items():
            gt = ground_truths.get(id_, "")
            if isinstance(gt, str):
                gt_text = gt.strip()
            else:
                # Nếu là list, lấy phần tử đầu tiên
                gt_text = gt[0].strip() if gt else ""
            
            pred_text = pred.strip()
            
            # Chỉ tính score cho các cặp không rỗng
            if gt_text and pred_text:
                # Text đã được normalized rồi, không cần .lower() lại
                gts[id_] = [gt_text]
                res[id_] = [pred_text]
                valid_ids.append(id_)
        
        if not valid_ids:
            return {id_: 0.0 for id_ in predictions.keys()}
        
        try:
            avg_score, individual_scores = self.rouge_scorer.compute_score(gts, res)
            # individual_scores là list theo thứ tự của gts/res keys
            rouge_dict = {id_: score for id_, score in zip(valid_ids, individual_scores)}
            
            # Fill 0.0 cho các IDs không valid
            for id_ in predictions.keys():
                if id_ not in rouge_dict:
                    rouge_dict[id_] = 0.0
                    
            return rouge_dict
        except Exception as e:
            print(f"Error calculating ROUGE-L: {e}")
            import traceback
            traceback.print_exc()
            return {id_: 0.0 for id_ in predictions.keys()}

    def accuracy_reward(self, completion: str, solution: str) -> float:
        """
        Single sample version - tính hybrid reward cho một sample.
        """
        # Extract answers
        sol_match = re.search(r"<CONCLUSION>(.*?)</CONCLUSION>", solution, flags=re.DOTALL | re.IGNORECASE)
        ground_truth = sol_match.group(1).strip() if sol_match else solution.strip()

        content_match = re.search(r"<CONCLUSION>(.*?)</CONCLUSION>", completion, flags=re.DOTALL | re.IGNORECASE)
        student_answer = content_match.group(1).strip() if content_match else ""

        # Apply cleaning và normalization
        ground_truth_cleaned = normalize_answer(clean_text(ground_truth))
        student_answer_cleaned = normalize_answer(clean_text(student_answer))

        # Nếu sau normalize mà rỗng -> phạt -1
        if not student_answer_cleaned:
            return -1.0
            
        # Tính cả hai metrics với text đã cleaned
        gts_dict = {0: ground_truth_cleaned}
        preds_dict = {0: student_answer_cleaned}
        
        bert_score = self.calculate_bertscore_batch(gts_dict, preds_dict).get(0, 0.0)
        rouge_score = self.calculate_rouge_batch(gts_dict, preds_dict).get(0, 0.0)
    
        # Combined reward
        reward = self.alpha * bert_score + (1.0 - self.alpha) * rouge_score
        
        # Nếu reward nhỏ hơn threshold, gán -1
        if reward < self.threshold:
            return -1.0
        
        return reward

    def accuracy_rewards_batch(self, completions: list[str], solutions: list[str]) -> list[float]:
        """
        Batch version - trả về hybrid reward (ROUGE-L + BERTScore) cho từng sample (0-1).
        """
        if not completions:
            return []
        
        # Extract tất cả answers
        gts_dict = {}
        preds_dict = {}
        empty_indices = set()  # Track các index có answer rỗng
        
        for i, (completion, solution) in enumerate(zip(completions, solutions)):
            sol_match = re.search(r"<CONCLUSION>(.*?)</CONCLUSION>", solution, flags=re.DOTALL | re.IGNORECASE)
            ground_truth = sol_match.group(1).strip() if sol_match else solution.strip()

            content_match = re.search(r"<CONCLUSION>(.*?)</CONCLUSION>", completion, flags=re.DOTALL | re.IGNORECASE)
            student_answer = content_match.group(1).strip() if content_match else ""
            
            # Apply cleaning và normalization trước khi tính metrics
            ground_truth_cleaned = normalize_answer(clean_text(ground_truth))
            student_answer_cleaned = normalize_answer(clean_text(student_answer))
            
            # Nếu sau normalize mà rỗng -> đánh dấu để phạt
            if not student_answer_cleaned:
                empty_indices.add(i)
            else:
                gts_dict[i] = ground_truth_cleaned
                preds_dict[i] = student_answer_cleaned
        
        # Tính cả ROUGE-L và BERTScore batch với cleaned text (chỉ cho non-empty)
        rouge_scores = self.calculate_rouge_batch(gts_dict, preds_dict)
        bert_scores = self.calculate_bertscore_batch(gts_dict, preds_dict)
        
        # Clear cache để tránh memory leak
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Kết hợp scores với trọng số alpha
        rewards = []
        for i in range(len(completions)):
            if i in empty_indices:
                # Phạt -1 cho answer rỗng
                reward = -1.0
                print(f"  [Sample {i}] Empty answer -> Penalty={reward}")
            else:
                rouge_score = rouge_scores.get(i, 0.0)
                bert_score = bert_scores.get(i, 0.0)
                
                # Hybrid reward
                reward = self.alpha * bert_score + (1.0 - self.alpha) * rouge_score
                
                # Nếu reward nhỏ hơn threshold, gán -1
                if reward < self.threshold:
                    print(f"  [Sample {i}] ROUGE-L={rouge_score:.4f}, BERTScore={bert_score:.4f} -> Reward={reward:.4f} < {self.threshold} -> Penalty=-1.0")
                    reward = -1.0
                else:
                    print(f"  [Sample {i}] ROUGE-L={rouge_score:.4f}, BERTScore={bert_score:.4f} -> Reward={reward:.4f}")
            
            rewards.append(reward)
        
        return rewards


import json

class CaptionRewardScorer(BaseRewardScorer):
    """
    Caption reward scorer sử dụng BERTScore cho semantic matching với COCO captions.
    So sánh student caption với tất cả GT captions và lấy max BERTScore.
    Trả về điểm BERTScore trực tiếp (0.0 - 1.0).
    """
    
    def __init__(self, 
                 model_name_or_path="bert",
                 coco_train_path="/home/vlai-vqa-nle/minhtq/vqa-nle/data/processed/coco/coco_train2014_with_captions.json",
                 coco_val_path="/home/vlai-vqa-nle/minhtq/vqa-nle/data/processed/coco/coco_val2014_with_captions.json"):
        """
        Args:
            model_name_or_path: Đường dẫn đến BERT model.
            coco_train_path: Đường dẫn đến COCO train captions JSON
            coco_val_path: Đường dẫn đến COCO val captions JSON
        """
        self.model_path = model_name_or_path
        # Khởi tạo shared BERTScore một lần
        self.initialize_bertscore(self.model_path)
        
        # Load COCO captions một lần
        print("Loading COCO captions...")
        self.coco_captions = {}
        
        # Load train captions
        with open(coco_train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
            self.coco_captions.update(train_data['captions_map'])
        
        # Load val captions
        with open(coco_val_path, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
            self.coco_captions.update(val_data['captions_map'])
        
        print(f"Loaded {len(self.coco_captions)} images with captions")
    
    def caption_rewards_batch(self, completions: list[str], solutions: list[str]) -> list[float]:
        """
        Batch version - trả về max BERTScore F1 cho từng sample (0-1).
        So sánh student caption với tất cả GT captions của image.
        """
        if not completions:
            return []
        
        rewards = []
        
        for completion, solution in zip(completions, solutions):
            # Extract image_id từ solution
            sol_match = re.search(r"<CAPTION>(.*?)</CAPTION>", solution, flags=re.DOTALL | re.IGNORECASE)
            image_id_str = sol_match.group(1).strip() if sol_match else solution.strip()
            
            
            # Extract student caption từ completion
            content_match = re.search(r"<CAPTION>(.*?)</CAPTION>", completion, flags=re.DOTALL | re.IGNORECASE)
            student_caption = content_match.group(1).strip() if content_match else ""
            
            if not student_caption:
                rewards.append(0.0)
                continue
            
            # Lấy tất cả GT captions từ COCO
            if image_id_str not in self.coco_captions:
                print(f"Warning: Image ID {image_id_str} not found in COCO captions")
                rewards.append(0.0)
                continue
            
            gt_captions_list = self.coco_captions[image_id_str]  # List of dicts with 'caption' and 'caption_id'
            gt_captions = [item['caption'] for item in gt_captions_list]
            
            # Tính BERTScore với từng GT caption
            gts_dict = {j: gt_cap for j, gt_cap in enumerate(gt_captions)}
            preds_dict = {j: student_caption for j in range(len(gt_captions))}
            
            bert_scores = self.calculate_bertscore_batch(
                gts_dict, 
                preds_dict,
                model_name_or_path=self.model_path
            )
            
            # Lấy max score
            max_score = max(bert_scores.values()) if bert_scores else 0.0
            rewards.append(max_score)
        
        return rewards