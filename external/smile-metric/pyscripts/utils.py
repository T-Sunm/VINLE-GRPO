import time
import json
import bert_score
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# specific to METEOR Implementation
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
nltk.download('wordnet')

def compute_rouge_score(metrics:list=['rougeL'], pred_col='pred', sub_metrics=['fmeasure'], ref_data=None, ans_idx:int=1):
    """
    Computes ROUGE scores between reference and candidate sentences.

    Parameters:
        metrics (list): List of ROUGE metrics to compute (e.g., ['rouge1', 'rouge2','rougeL']).
        pred_col (str): Name of the prediction column (default: 'pred').
        sub_metrics (list): List of sub-metrics to extract (e.g., ['fmeasure']).
        ref_data (list): List of data samples, each containing answer and prediction.
        ans_idx (int): Index of the answer to use (1 for actual answer, 2 for synthetic answer).

    Returns:
        dict: Dictionary of ROUGE scores for each metric and sub-metric.
    """
    ans, preds = [], []
    for data in ref_data:
        # index - ans_idx is the answer, last index is the prediction
        ans.append(str(data[ans_idx]))
        preds.append(data[-1])

    # Initialize ROUGE scorer
    # egs - ['rouge1', 'rouge2', 'rougeL']
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
    rouge_rslts = {metric: {sub_metric:[] for sub_metric in sub_metrics} for metric in metrics}

    for ref, cand in tqdm(zip(ans, preds), total=len(ans)):
        scores = scorer.score(ref, cand)
        for key, data in rouge_rslts.items():
            for metric in sub_metrics:
                if metric=='fmeasure':
                    data[metric].append(scores[key].fmeasure)

    
    return rouge_rslts

def compute_bert_score(inp_data, pred_col='pred', ans_idx:int=1):
    """
    Computes BERTScore precision, recall, and F1 between reference and prediction strings.

    Parameters:
        inp_data (list): List of data samples, each containing answer and prediction.
        pred_col (str): Name of the prediction column (default: 'pred').
        ans_idx (int): Index of the answer to use (1 for actual answer, 2 for synthetic answer).

    Returns:
        dict: Dictionary with BERTScore precision ('P'), recall ('R'), and F1 ('F1') lists.
    """
    # Extract ans & pred
    # index-1 is the 'answer', last index is the prediction
    ans = [str(data[ans_idx]) for data in inp_data]
    pred = [str(data[-1]) for data in inp_data]
    
    bert_p, bert_r, bert_f1 = bert_score.score(pred, ans, lang='en')
    
    bert_result = {'P':[], 'R':[], 'F1': []}
    for p,r,f1 in zip(bert_p, bert_r, bert_f1):
        bert_result['P'].append(p.item())
        bert_result['R'].append(r.item())
        bert_result['F1'].append(f1.item())

    return bert_result

def compute_meteor_score(inp_data, pred_col='pred', ans_idx:int=1):
    """
    Calculates the METEOR score between a reference and prediction text.

    Args:
        inp_data (list): List of data samples, each containing answer and prediction.
        pred_col (str): Name of the prediction column (default: 'pred').
        ans_idx (int): Index of the answer to use (1 for actual answer, 2 for synthetic answer).

    Returns:
        dict: Dictionary with METEOR scores ('meteor').
    """
    ans = [str(data[ans_idx]) for data in inp_data]
    preds = [str(data[-1]) for data in inp_data]

    result = {'meteor':[]}
    for ref, cand in tqdm(zip(ans, preds), total=len(ans)):
        tokenized_reference = word_tokenize(ref)
        tokenized_hypothesis = word_tokenize(cand)
        result['meteor'].append(meteor_score([tokenized_reference], tokenized_hypothesis))
    
    return result

def compute_exact_match(inp_data, pred_col='pred', ans_idx:int=1):
    """
    Computes the exact match(after lowercasing) between reference and prediction strings.

    Parameters:
        inp_data (list): List of data samples, each containing answer and prediction.
        pred_col (str): Name of the prediction column (default: 'pred').
        ans_idx (int): Index of the answer to use (1 for actual answer, 2 for synthetic answer).

    Returns:
        dict: Dictionary with exact match results ('exact_match'), 1 if exact match, else 0.
    """
    ans = [str(data[ans_idx]) for data in inp_data]
    preds = [str(data[-1]) for data in inp_data]

    result = {'exact_match':[]}
    for ref, cand in tqdm(zip(ans, preds), total=len(ans)):
        tokenized_reference = ref.lower() if not ref.isdigit() else ref
        tokenized_hypothesis = cand.lower() if not cand.isdigit() else cand
        result['exact_match'].append(int(tokenized_reference == tokenized_hypothesis))
    
    return result

def compute_sbert_score(inp_data, ans_idx:int=1):
    """
    Computes cosine similarity between sentence embeddings of reference and prediction strings using SBERT.

    Parameters:
        inp_data (list): List of data samples, each containing answer and prediction.
        ans_idx (int): Index of the answer to use (1 for actual answer, 2 for synthetic answer).

    Returns:
        np.ndarray: Array of cosine similarity scores for each sample.
    """
    ans = [str(data[ans_idx]) for data in inp_data]
    preds = [str(data[-1]) for data in inp_data]

    # Initialise sbert
    # Change the SBERT model here accordingly
    model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
    ans_embs = model.encode(ans)
    pred_embs = model.encode(preds)

    # Generate cosine-similarities
    sims = np.diagonal(cosine_similarity(ans_embs, pred_embs))

    return sims

def compute_bleurt_score(inp_data, ans_idx=1, checkpoint='BLEURT-20'):
    """
    Computes BLEURT scores between reference and prediction strings using Google's learned metric.

    Parameters:
        inp_data (list): List of data samples, each containing answer and prediction.
        ans_idx (int): Index of the answer to use (1 for actual answer, 2 for synthetic answer).
        checkpoint (str): BLEURT checkpoint to use (default: 'BLEURT-20'). 
                         Options: 'BLEURT-20', 'BLEURT-20-D12', 'BLEURT-20-D6' (smaller, faster).

    Returns:
        dict: Dictionary with 'scores' key containing list of BLEURT scores for each sample.
    """
    from evaluate import load
    
    ans = [str(data[ans_idx]) for data in inp_data]
    preds = [str(data[-1]) for data in inp_data]

    # Initialize BLEURT scorer
    # Using the evaluate library (from Hugging Face)
    bleurt = load('bleurt', checkpoint)
    
    # Compute scores in batches for efficiency
    scores = []
    batch_size = 32  # Process in batches to avoid memory issues
    
    for i in tqdm(range(0, len(ans), batch_size), desc="Computing BLEURT"):
        batch_refs = ans[i:i+batch_size]
        batch_preds = preds[i:i+batch_size]
        batch_scores = bleurt.compute(predictions=batch_preds, references=batch_refs)
        scores.extend(batch_scores['scores'])
    
    return {'scores': scores}

def compute_moverscore(inp_data, ans_idx=1, model='distilbert-base-uncased', n_gram=2, device='cuda'):
    """
    Computes MoverScore between reference and prediction strings using contextualized embeddings and Word Mover's Distance.
    
    Based on Zhao et al. (EMNLP 2019), recommended configuration for QA tasks:
    - model: 'bert-base-uncased' or BERT fine-tuned on MNLI for best correlation with human judgments
    - n_gram: 2 (bigrams) - captures phrase-level context and word order, shown to outperform unigrams
    - For QA evaluation, bigrams are particularly effective as they capture multi-word answer phrases

    Parameters:
        inp_data (list): List of data samples, each containing answer and prediction.
        ans_idx (int): Index of the answer to use (1 for actual answer, 2 for synthetic answer).
        model (str): Model to use for embeddings (default: 'distilbert-base-uncased').
                    Recommended for QA: 'bert-base-uncased' (best quality, slower) or 
                    'distilbert-base-uncased' (faster, slightly lower quality).
                    Paper's best: BERT fine-tuned on MNLI dataset.
                    Other options: 'roberta-base', 'roberta-large', 'albert-base-v2'.
        n_gram (int): N-gram level for matching (default: 2).
                     1 = unigrams (individual words)
                     2 = bigrams (word pairs, recommended for QA - captures phrases like "New York")
                     3 = trigrams (better for longer contexts)
        device (str): Device to use for computation (default: 'cuda').

    Returns:
        dict: Dictionary with 'scores' key containing list of MoverScore values for each sample.
        
    References:
        Zhao et al. (2019). MoverScore: Text Generation Evaluating with Contextualized 
        Embeddings and Earth Mover Distance. EMNLP 2019.
        Paper findings: n_gram=2 with BERT-MNLI achieved highest correlation with human judgments.
    """
    try:
        from moverscore_v2 import get_idf_dict, word_mover_score
    except ImportError:
        raise ImportError(
            "MoverScore not found. Please install it using:\n"
            "pip install -U git+https://github.com/AIPHES/emnlp19-moverscore.git"
        )
    
    # Set the model via environment variable (MoverScore's way of selecting models)
    os.environ['MOVERSCORE_MODEL'] = model
    
    ans = [str(data[ans_idx]) for data in inp_data]
    preds = [str(data[-1]) for data in inp_data]

    # Compute IDF dictionary for references (used for weighting)
    idf_dict_ref = get_idf_dict(ans)
    idf_dict_hyp = get_idf_dict(preds)
    
    # Compute MoverScore
    # The function returns scores for each reference-hypothesis pair
    # n_gram=2 recommended by paper for better correlation with human judgments in QA tasks
    scores = word_mover_score(
        ans, 
        preds, 
        idf_dict_ref, 
        idf_dict_hyp,
        stop_words=[], 
        n_gram=n_gram, 
        remove_subwords=True,
        batch_size=32,
        device=device
    )
    
    return {'scores': scores}

def time_exec(start_time, end_time, title):
    """
    Prints the elapsed time between start_time and end_time with a custom title.

    Parameters:
        start_time (float): Start time in seconds (as returned by time.time()).
        end_time (float): End time in seconds (as returned by time.time()).
        title (str): Description of the timed operation.
    """
    elapsed_time = end_time - start_time
    print(f' > {title}: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

def read_json(inp_file):
    """
    Reads a JSON or JSONL file and returns its contents.

    Parameters:
        inp_file (str): Path to the input JSON or JSONL file.

    Returns:
        object: Parsed data from the file.
    """
    if inp_file[-1] == 'l':
        # if it a .jsonl file
        with open(inp_file,'r') as f:
            inp_data = [json.loads(line) for line in f]
    else:
        inp_data = json.load(open(inp_file))

    return inp_data