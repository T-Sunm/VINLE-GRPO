from typing import Dict, List
import re
from underthesea import pos_tag, text_normalize, word_tokenize


class ReasoningRewardScorer:
    """
    Reward scorer với preprocessing: text_normalize + word_tokenize
    """

    QUESTION_TYPE_TO_CLUSTER = {
        # Nhóm 1: Trạng thái/Tính chất
        "are the": 1, "are they": 1, "has": 1, "is he": 1, "is it": 1,
        "is the": 1, "is the person": 1, "is there": 1, "is there a": 1,
        "is this person": 1, "was": 1,

        # Nhóm 2: What xác định cụ thể
        "what animal is": 2, "what brand": 2, "what color": 2, "what color is": 2,
        "what kind of": 2, "what room is": 2, "what sport is": 2, "what time": 2,
        "what type of": 2, "which": 2,

        # Nhóm 3: What mở
        "can you": 3, "could": 3, "how": 3, "what are": 3,
        "what is the person": 3, "what is this": 3, "what number is": 3,
        "where are the": 3,

        # Nhóm 4: Câu hỏi thông thường
        "are": 4, "are there": 4, "are these": 4, "do": 4, "do you": 4,
        "does the": 4, "does this": 4, "is": 4, "is the man": 4,
        "is the woman": 4, "is this": 4, "none of the above": 4,
        "what are the": 4, "what does the": 4, "what is": 4,
        "where is the": 4, "who is": 4,

        # Nhóm 5: Nhận dạng với ADJ cao
        "is that a": 5, "is this a": 5, "is this an": 5,

        # Nhóm 6: What về người/đối tượng
        "what": 6, "what is the": 6, "what is the man": 6, "what is the woman": 6,
    }

    CLUSTER_WEIGHTS = {
        1: {"NOUN": 0.30, "VERB": 0.50, "ADJ": 0.20},
        2: {"NOUN": 0.70, "VERB": 0.20, "ADJ": 0.10},
        3: {"NOUN": 0.50, "VERB": 0.30, "ADJ": 0.20},
        4: {"NOUN": 0.45, "VERB": 0.40, "ADJ": 0.15},
        5: {"NOUN": 0.35, "VERB": 0.30, "ADJ": 0.35},
        6: {"NOUN": 0.60, "VERB": 0.30, "ADJ": 0.10},
    }

    NOUN_TAGS = ["N", "Np"]
    VERB_TAGS = ["V", "Vy"]
    ADJ_TAGS = ["A"]

    def __init__(self, threshold: float = 0.3, use_preprocessing: bool = True):
        """
        Args:
            threshold: Ngưỡng tối thiểu cho reward
            use_preprocessing: Bật/tắt text normalization và word tokenization
        """
        self.threshold = threshold
        self.use_preprocessing = use_preprocessing

    def preprocess_text(self, text: str) -> str:
        """
        Preprocessing: text_normalize + word_tokenize

        Args:
            text: Raw text

        Returns:
            Preprocessed text
        """
        if not text or not text.strip():
            return ""

        try:
            return text_normalize(text)
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return text

    def extract_pos_tags(self, text: str) -> Dict[str, set]:
        """
        Trích xuất POS tags sau khi preprocessing
        """
        if not text or not text.strip():
            return {"NOUN": set(), "VERB": set(), "ADJ": set(), "OTHERS": set()}

        try:
            # Apply preprocessing nếu được bật
            if self.use_preprocessing:
                text = self.preprocess_text(text)

            # POS tagging
            tagged = pos_tag(text.lower().strip())

            pos_dict = {
                "NOUN": set(),
                "VERB": set(),
                "ADJ": set(),
                "OTHERS": set()
            }

            for word, tag in tagged:
                # Loại bỏ underscore từ word_tokenize nếu có
                word_clean = word.replace("_", " ")

                if tag in self.NOUN_TAGS:
                    pos_dict["NOUN"].add(word_clean)
                elif tag in self.VERB_TAGS:
                    pos_dict["VERB"].add(word_clean)
                elif tag in self.ADJ_TAGS:
                    pos_dict["ADJ"].add(word_clean)
                else:
                    pos_dict["OTHERS"].add(word_clean)

            return pos_dict
        except Exception as e:
            print(f"Error in POS tagging: {e}")
            return {"NOUN": set(), "VERB": set(), "ADJ": set(), "OTHERS": set()}

    def extract_object_extraction(self, solution: str) -> set:
        """
        Trích xuất danh sách object từ thẻ <object_extraction>
        
        Args:
            solution: Ground truth chứa thẻ <object_extraction>
        
        Returns:
            set: Tập hợp các object đã được normalize
        """
        obj_match = re.search(
            r"<object_extraction>(.*?)</object_extraction>",
            solution,
            flags=re.DOTALL | re.IGNORECASE
        )
        
        if not obj_match:
            return set()
        
        obj_text = obj_match.group(1).strip()
        
        # Split by comma và normalize từng object
        objects = set()
        for obj in obj_text.split(','):
            obj_clean = obj.strip()
            if obj_clean:

                if self.use_preprocessing:
                    obj_clean = self.preprocess_text(obj_clean)
                objects.add(obj_clean.lower())
        
        return objects

    def jaccard_similarity(self, set1: set, set2: set) -> float:
        """Tính Jaccard similarity"""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def calculate_weighted_jaccard(
        self,
        reasoning_pos: Dict[str, set],
        explain_pos: Dict[str, set],
        weights: Dict[str, float]
    ) -> float:
        """Tính weighted Jaccard với dynamic weighting - chỉ tính các POS có trong GT"""
        
        # Xác định các POS types có trong explain (GT)
        active_pos_types = [pos_type for pos_type in ["NOUN", "VERB", "ADJ"] 
                            if len(explain_pos[pos_type]) > 0]
        
        # Nếu explain rỗng hoàn toàn
        if not active_pos_types:
            return 0.0
        
        # Tính tổng weight của các active POS types
        total_active_weight = sum(weights[pos_type] for pos_type in active_pos_types)
        
        # Tính weighted score chỉ cho active POS types
        weighted_score = 0.0
        for pos_type in active_pos_types:
            jaccard = self.jaccard_similarity(reasoning_pos[pos_type], explain_pos[pos_type])
            # Normalize weight theo tổng active weight
            normalized_weight = weights[pos_type] / total_active_weight
            weighted_score += normalized_weight * jaccard
        
        return weighted_score

    def get_cluster_for_question_type(self, question_type: str) -> int:
        """Lấy cluster ID cho question type"""
        qt_lower = question_type.lower().strip()
        return self.QUESTION_TYPE_TO_CLUSTER.get(qt_lower, 4)

    def reasoning_reward(self, completion: str, solution: str) -> float:
        """
        Tính reward cho 1 sample với preprocessing
        """
        # Extract question type
        qt_match = re.search(
            r"<questiontype>(.*?)</questiontype>",
            solution,
            flags=re.DOTALL | re.IGNORECASE
        )
        question_type = qt_match.group(1).strip() if qt_match else ""

        # Extract explain (ground truth)
        explain_match = re.search(
            r"<explain>(.*?)</explain>",
            solution,
            flags=re.DOTALL | re.IGNORECASE
        )
        explain_text = explain_match.group(1).strip() if explain_match else ""

        # Extract reasoning (prediction)
        reasoning_match = re.search(
            r"<REASONING>(.*?)</REASONING>",
            completion,
            flags=re.DOTALL | re.IGNORECASE
        )
        reasoning_text = reasoning_match.group(1).strip() if reasoning_match else ""

        # Validation
        if not reasoning_text:
            return -1.0

        if not explain_text:
            return 0.0

        # Get cluster and weights
        cluster_id = self.get_cluster_for_question_type(question_type)
        weights = self.CLUSTER_WEIGHTS[cluster_id]

        # POS tagging (với preprocessing)
        reasoning_pos = self.extract_pos_tags(reasoning_text)
        explain_pos = self.extract_pos_tags(explain_text)

        # Calculate weighted Jaccard
        reward = self.calculate_weighted_jaccard(reasoning_pos, explain_pos, weights)

        # Apply threshold
        # if reward < self.threshold:
        #     return -1.0

        return reward
        
    def reasoning_reward_with_object_extraction(self, completion: str, solution: str) -> float:
        """
        Tính reward dựa trên so khớp giữa danh từ trong reasoning và object_extraction
        
        Args:
            completion: Output từ model (chứa <REASONING>)
            solution: Ground truth (chứa <object_extraction>)
        
        Returns:
            float: Reward score
        """
        # Extract reasoning (prediction)
        reasoning_match = re.search(
            r"<REASONING>(.*?)</REASONING>",
            completion,
            flags=re.DOTALL | re.IGNORECASE
        )
        reasoning_text = reasoning_match.group(1).strip() if reasoning_match else ""
        
        # Validation
        if not reasoning_text:
            return -1.0
        
        # Extract noun_count từ GT
        noun_count_match = re.search(
            r"<noun_count>(\d+)</noun_count>",
            solution,
            flags=re.DOTALL | re.IGNORECASE
        )
        gt_noun_count = int(noun_count_match.group(1)) if noun_count_match else 0
        
        # Extract object_extraction
        gt_objects = self.extract_object_extraction(solution)
        if len(gt_objects) < gt_noun_count:
            reference_count = len(gt_objects)
        else:
            reference_count = gt_noun_count
        
        if reference_count == 0:
            return 0.0
        
        # POS tagging để lấy danh từ từ reasoning
        reasoning_pos = self.extract_pos_tags(reasoning_text)
        reasoning_nouns = reasoning_pos["NOUN"]
        
        overlap = len(reasoning_nouns & gt_objects)
        reward = overlap / reference_count
        reward = min(reward, 1.0)
        
        return reward
        
    def reasoning_rewards_batch(
        self,
        completions: List[str],
        solutions: List[str]
    ) -> List[float]:
        """
        Batch version với logging
        """
        if not completions:
            return []

        rewards = []
        for i, (completion, solution) in enumerate(zip(completions, solutions)):
            reward = self.reasoning_reward_with_object_extraction(completion, solution)

            if reward == -1.0:
                print(f"  [Sample {i}] Empty reasoning or below threshold -> Penalty={reward}")
            else:
                qt_match = re.search(r"<questiontype>(.*?)</questiontype>", solution, re.DOTALL | re.IGNORECASE)
                qt = qt_match.group(1).strip() if qt_match else "unknown"
                cluster = self.get_cluster_for_question_type(qt)
                print(f"  [Sample {i}] QuestionType={qt} (Cluster {cluster}) -> Reward={reward:.4f}")

            rewards.append(reward)

        return rewards