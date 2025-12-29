import os
import argparse
import json
import numpy as np
import time
import pickle as pkl
from smile.smile import SMILE
from utils import compute_rouge_score, compute_bert_score, compute_meteor_score, compute_exact_match, compute_sbert_score, compute_bleurt_score, compute_moverscore
import sys
from tqdm import tqdm
sys.path.append('..')

def parse_arguments():
    """
    Parses command-line arguments for the score generation script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Add all the arguments
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input file.')
    parser.add_argument('--pred_file', type=str, help="Path to the prediction file(useful when you don't have syn_ans in same file).")
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output file.')
    parser.add_argument('--eval_mode', choices=['smile', 'rouge', 'bert_score', 'meteor', 'exact_match', 'sbert', 'bleurt', 'moverscore'], default='smile', help='Which scores to generate.')
    parser.add_argument("--timeit", action="store_true", help="Enable timing of the code execution.")
    parser.add_argument("--emb_model", default="ember-v1", type=str, help="Embedding model to use with SMILE.")
    parser.add_argument("--save_emb_folder", type=str, help="Saves the embedding from a specified folder, generates 4 files i.e. <dataset>/<syn and model>/syn_ans_emb.npy, <dataset>/<pred model>/pred_sent_emb.npy, <dataset>/ans_kwd_emb.npy & <dataset>/<pred model>pred_kwd_emb.npy.")
    parser.add_argument("--load_emb_folder", type=str, help="Loads the embedding from a specified folder, expects atleast 2 files i.e. syn_ans_emb.npy, ans_kwd_emb.npy. If pred emb files are not present, it creates it on the fly.")
    parser.add_argument('--syn_ans_model', type=str, help='Model used for generating "synthetic answers", required with --save_emb_folder, as it maps syn_ans_emb to this model.')
    parser.add_argument('--use_ans', action='store_true', help="Uses answers in place of 'syn_ans' for evaluation.")
    parser.add_argument("--verbose", action="store_true", help="Runs the evaluation in the verbose mode.")

    args = parser.parse_args()
    return args

def save_dict(data, file_path):
    """
    Saves a Python object to a file using pickle.

    Parameters:
        data (object): The data to save.
        file_path (str): Path to the output file.
    """
    with open(file_path, 'wb') as f:
        pkl.dump(data, f)

def read_json(inp_file):
    """
    Reads a JSON or JSONL file and returns its contents.

    Parameters:
        inp_file (str): Path to the input JSON or JSONL file.

    Returns:
        object: Parsed data from the file.
    """
    if inp_file[-1] == 'l':
        with open(inp_file,'r') as f:
            inp_data = [json.loads(line) for line in f]
    else:
        inp_data = json.load(open(inp_file))

    return inp_data

def format_data(inp_data, pred_data, use_ans=False):
    """
    Formats input and prediction data into the expected evaluation format.

    Parameters:
        inp_data (list): List of input data dictionaries(should contain keys - id/ question_id, question, answer, syn_ans(optional), pred).
        pred_data (list or None): List of prediction data dictionaries, or None.
        use_ans (bool): If True, use 'answer' instead of 'syn_ans' for evaluation.

    Returns:
        np.ndarray or list: Formatted data ready for SMILE evaluation.
    """
    proc_data = []
    # Extract the data in expected format
    if not pred_data:
        # If the same inp_data file contains prediction
        for data in inp_data:
            proc_data.append((
            data['question'],
            data['answer'],
            data.get('answer' if use_ans else 'syn_ans', ''),
            data['pred']
        ))
    else:
        # If the inp_data and prediction files are separate
        for inp, pred in zip(inp_data, pred_data):
            inp_qid = 'id' if 'id' in inp.keys() else 'question_id'
            pred_qid = 'id' if 'id' in pred.keys() else 'question_id'
            if inp[inp_qid]!=pred[pred_qid]:
                raise("Order of the data is not correct, please reorder the data")
            proc_data.append((
            inp['question'],
            inp['answer'],
            inp['answer' if use_ans else 'syn_ans'],
            pred['pred']
        ))
    # Output the size of the extracted data
    if isinstance(inp_data[0]['answer'], str) : 
        proc_data = np.array(proc_data)
        print(f' > input data size : {proc_data.shape}')
    elif isinstance(inp_data[0]['answer'], list):
        # If answer contains multiple reference answers
        print(f' > input data size : {len(proc_data)}')
    return proc_data

def load_data(args):
    """
    Loads and formats input and prediction data based on command-line arguments.

    Parameters:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        np.ndarray or list: Formatted data ready for evaluation.
    """
    inp_data = read_json(args.input_file)
    pred_data = read_json(args.pred_file) if args.pred_file else None
    
    # Check if the pred_path is present
    use_ans = args.use_ans if args.use_ans else False
    proc_data = format_data(inp_data, pred_data, use_ans)

    return proc_data

def main():
    """
    Main entry point for the evaluation script

    Handles argument parsing, data loading, evaluation mode selection, score computation,
    timing, and saving of results.

    Steps:
        1. Parses command-line arguments.
        2. Loads and formats input and prediction data.
        3. Selects and runs the specified evaluation mode (SMILE, ROUGE, BERTScore, METEOR, Exact Match, SBERT).
        4. Optionally prints timing information.
        5. Saves the computed results to the specified output file.
    """
    # Parse cmd line arguments
    args = parse_arguments()

    # open the input file
    print('1. Loading input files in specific format')
    proc_data = load_data(args)

    # Use index 1 for actual answer if use_ans is True, else use index 2 for synthetic answer
    ans_idx = 1 if args.use_ans else 2

    # Load smile object and perform the evaluation
    # Generate smile scores
    if args.eval_mode == 'smile':
        print('2. Loading SMILE')
        # Choose the metrics to be computed - supports 'avg', 'hm'
        eval_metrics = ['avg', 'hm']
        smile_obj = SMILE(args.emb_model,
                          eval_metrics, 
                          assign_bins=True, 
                          use_exact_matching=True, 
                          save_emb_folder=args.save_emb_folder, 
                          load_emb_folder=args.load_emb_folder, 
                          syn_ans_model=args.syn_ans_model, 
                          verbose=args.verbose)

        print('3. Generate SMILE score')
        start_time = time.time()
        # When syn_ans/ answer is a list, rather than a string.
        if isinstance(proc_data, list):
            # Iterate over each ans/ syn_ans and generate score.
            results = {'sent_emb_scores': [[]], 'kwd_emb_scores': [[]], 'max_sim_words': [[]], 'frac_exact_match': [[]], 'kwd_scores':[[]]}
            for m in eval_metrics:
                results[f'{m} score'] = [[]]
                results[m]=[[]]
            for i, sample in enumerate(tqdm(proc_data, total=len(proc_data))):
                # answer & syn_ans are always present at index 1 and 2 respectively.
                for a, syn_a in zip(sample[1], sample[2]):
                    # Generate scores for each answer and syn_ans
                    single_sample = np.array([sample[0], a, syn_a, sample[3]]).reshape(1,-1)
                    single_result = smile_obj.generate_scores(single_sample)
                    # Update the results
                    for k, v in single_result.items():
                        if k in results:
                            if k=='max_sim_words':
                                results[k][-1].extend(v)
                            else:
                                results[k][-1].append(v)
                
                # Append a new empty list for all the keys in results
                if i < len(proc_data)-1:
                    for k in results.keys():
                        results[k].append([])

        else:
            # When syn_ans/ answer is a string
            results = smile_obj.generate_scores(proc_data)

        end_time = time.time()
        if args.timeit: 
            elapsed_time = end_time - start_time
            print(f' > SMILE evaluation time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')
    
    elif args.eval_mode == 'rouge':
        print('2. Loading ROUGE')
        print('3. Generate ROUGE score')
        start_time = time.time()
        # Hardcoding rouge parameters
        # Change values here
        results = compute_rouge_score(metrics=['rougeL'], pred_col='pred', sub_metrics=['fmeasure'],ref_data=proc_data, ans_idx=ans_idx)
        end_time = time.time()
        if args.timeit: 
            elapsed_time = end_time - start_time
            print(f' > ROUGE evaluation time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

    elif args.eval_mode == 'bert_score':
        print('2. Loading BERT Scorer')
        print('3. Generate BERT score')
        start_time = time.time()
        results = compute_bert_score(inp_data=proc_data, pred_col='pred', ans_idx=ans_idx)
        end_time = time.time()
        if args.timeit: 
            elapsed_time = end_time - start_time
            print(f' > BERTScore evaluation time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

    elif args.eval_mode == 'meteor':
        print('2. Loading METEOR Scorer')
        print('3. Generate METEOR score')
        start_time = time.time()
        results = compute_meteor_score(inp_data=proc_data, pred_col='pred', ans_idx=ans_idx)
        end_time = time.time()
        if args.timeit: 
            elapsed_time = end_time - start_time
            print(f' > METEOR Score evaluation time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')
    
    elif args.eval_mode == 'exact_match':
        print("2. Generating 'exact_match' scores")
        start_time = time.time()
        results = compute_exact_match(inp_data=proc_data, pred_col='pred', ans_idx=ans_idx)
        end_time = time.time()
        if args.timeit: 
            elapsed_time = end_time - start_time
            print(f' > Exact Match Score evaluation time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

    elif args.eval_mode == 'sbert':
        print("2. Generating 'sentence-bert' cosine scores")
        start_time = time.time()
        results = compute_sbert_score(inp_data=proc_data, ans_idx=ans_idx)
        end_time = time.time()
        if args.timeit: 
            elapsed_time = end_time - start_time
            print(f' > sBERT Score evaluation time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

    elif args.eval_mode == 'bleurt':
        print("2. Loading BLEURT Scorer")
        print("3. Generate BLEURT score")
        start_time = time.time()
        results = compute_bleurt_score(inp_data=proc_data, ans_idx=ans_idx)
        end_time = time.time()
        if args.timeit: 
            elapsed_time = end_time - start_time
            print(f' > BLEURT Score evaluation time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

    elif args.eval_mode == 'moverscore':
        print("2. Loading MoverScore")
        print("3. Generate MoverScore")
        start_time = time.time()
        model = 'bert-base-uncased'
        # For factual QA/ when using gt as is, '2' is recommended
        n_gram = 2 if args.use_ans else 3
        results = compute_moverscore(inp_data=proc_data, ans_idx=ans_idx, model=model, n_gram=n_gram)
        end_time = time.time()
        if args.timeit: 
            elapsed_time = end_time - start_time
            print(f' > MoverScore evaluation time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

    # Save the results
    # Ensure directory exists
    os.makedirs(f'{os.path.sep}'.join(args.output_file.split(os.path.sep)[:-1]), exist_ok=True)
    print('4. Saving the results')
    save_dict(results, args.output_file)
    print(f' > Results saved at -> {args.output_file}')

if __name__ == "__main__":
    main()