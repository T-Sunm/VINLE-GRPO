import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import nltk
import re
import os
import random
from tqdm import tqdm
import sys
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

# Download essential nltk vocabulary & tagger
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')
nltk.download('wordnet')

class PhoBERTWrapper:
    """
    Wrapper class for PhoBERT
    """
    def __init__(self, model_name, device):
        from transformers import RobertaModel, AutoTokenizer
        
        print(f"Loading PhoBERT from HuggingFace: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model = RobertaModel.from_pretrained(model_name)
        self.device = device
        self.model.to(device)
        self.model.eval()
        print(f"PhoBERT loaded successfully on {device}")
    
    def encode(self, sentences, batch_size=32, device=None, show_progress_bar=False, 
               normalize_embeddings=False, convert_to_numpy=True):
        if isinstance(sentences, str):
            sentences = [sentences]
        
        # Validate and sanitize inputs
        valid_sentences = []
        for s in sentences:
            if isinstance(s, str) and s.strip():
                # Remove non-BMP characters (emoji, special symbols)
                s_clean = ''.join(ch for ch in s if ord(ch) < 65536)
                # Remove null bytes
                s_clean = s_clean.replace('\x00', '').strip()
                if s_clean:
                    valid_sentences.append(s_clean)
                else:
                    valid_sentences.append(".")  # Placeholder for empty
            else:
                valid_sentences.append(".")
        
        all_embeddings = []
        iterator = range(0, len(valid_sentences), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Encoding")
        
        with torch.no_grad():
            for i in iterator:
                batch = valid_sentences[i:i+batch_size]
                
                try:
                    # Tokenize
                    encoded = self.tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=256,
                        return_tensors='pt'
                    )
                    
                    # Validate token IDs before moving to CUDA
                    input_ids = encoded['input_ids']
                    vocab_size = getattr(self.tokenizer, 'vocab_size', 64000)
                    
                    if (input_ids >= vocab_size).any() or (input_ids < 0).any():
                        print(f"Warning: Invalid token IDs in batch {i//batch_size}. Clamping to valid range.")
                        input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
                        encoded['input_ids'] = input_ids
                    
                    # Move to device
                    encoded = {k: v.to(self.device) for k, v in encoded.items()}
                    
                    # Get model output
                    outputs = self.model(**encoded)
                    
                    # Mean pooling
                    attention_mask = encoded['attention_mask']
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    
                    # Normalize if requested
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    
                    all_embeddings.append(embeddings.cpu())
                
                except (RuntimeError, torch.cuda.CudaError) as e:
                    print(f"CUDA error in batch {i//batch_size}: {e}")
                    print("Clearing CUDA cache and retrying on CPU...")
                    
                    # Clear CUDA cache
                    torch.cuda.empty_cache()
                    
                    try:
                        # Retry on CPU
                        cpu_device = torch.device('cpu')
                        encoded = self.tokenizer(
                            batch,
                            padding=True,
                            truncation=True,
                            max_length=256,
                            return_tensors='pt'
                        )
                        
                        # Clamp token IDs
                        vocab_size = getattr(self.tokenizer, 'vocab_size', 64000)
                        encoded['input_ids'] = torch.clamp(encoded['input_ids'], 0, vocab_size - 1)
                        
                        # Move model to CPU temporarily
                        original_device = next(self.model.parameters()).device
                        self.model.to(cpu_device)
                        
                        outputs = self.model(**encoded)
                        
                        attention_mask = encoded['attention_mask']
                        token_embeddings = outputs.last_hidden_state
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                        
                        if normalize_embeddings:
                            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                        
                        all_embeddings.append(embeddings)
                        
                        # Move model back to original device
                        self.model.to(original_device)
                        print(f"CPU fallback succeeded for batch {i//batch_size}")
                        
                    except Exception as cpu_error:
                        print(f"CPU fallback failed: {cpu_error}. Using zero embeddings for batch {i//batch_size}")
                        # Return zero embeddings as last resort
                        embedding_dim = 768
                        zero_emb = torch.zeros((len(batch), embedding_dim))
                        all_embeddings.append(zero_emb)
        
        # Concatenate all batches
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        if convert_to_numpy:
            return all_embeddings.numpy()
        return all_embeddings


class SMILE:
    def __init__(self, emb_model:str, eval_metrics:list, avg_w1=0.5, avg_w2=0.5, hm_w1=0.5, hm_w2=0.5, assign_bins=False, use_exact_matching=False, save_emb_folder=None, load_emb_folder=None, syn_ans_model=None, verbose=True):
        if torch.cuda.is_available():
            ## Uncomment the code below to randomly assign a gpu to use
            # cuda_visible_device = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            # gpu_list = cuda_visible_device.split(",")
            # selected_gpu = random.choice(gpu_list)
            self.device = torch.device(f"cuda")
        else:
            self.device = 'cpu'

        self.emb_model = self._get_emb_model(emb_model)
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.eval_metrics = eval_metrics
        self.avg_w1, self.avg_w2 = avg_w1, avg_w2
        self.hm_w1, self.hm_w2 = hm_w1, hm_w2
        self.assign_bins = assign_bins
        self.use_exact_matching = use_exact_matching
        self.save_emb_folder = save_emb_folder
        self.load_emb_folder = load_emb_folder
        self.verbose = verbose

        # Generate the paths to save/load the embedding
        # Also create the folder if not present
        if (self.save_emb_folder or self.load_emb_folder) and syn_ans_model:
            save_folder = self.save_emb_folder if self.save_emb_folder else self.load_emb_folder
            sep = os.path.sep
            self.SAVE_EMB_PATHS = {
                "syn_ans_emb": os.path.join(f'{sep}'.join(save_folder.split(sep)[:-1]), f"syn_ans_model-{syn_ans_model}{sep}syn_ans_emb.npy"),
                "pred_sent_emb":os.path.join(save_folder, 'pred_sent_emb.npy'),
                "ans_kwd_emb": os.path.join(f'{sep}'.join(save_folder.split(sep)[:-1]), "ans_kwd_emb.npy"),
                "pred_kwd_emb":os.path.join(save_folder, 'pred_kwd_emb.npy')
            }
            os.makedirs(save_folder, exist_ok=True)
            os.makedirs(os.path.join(f'{sep}'.join(save_folder.split(sep)[:-1]), f"syn_ans_model-{syn_ans_model}"), exist_ok=True)
        elif (self.save_emb_folder or self.load_emb_folder) and not syn_ans_model:
            if self.verbose: print('No valid syn_ans_model name found!')
            sys.exit(0)

        # Vectorize functions
        self.vect_process_kwds = np.vectorize(self._process_kwds)
        self.vect_get_pos_tag = np.vectorize(self._get_pos_tag)
        
    def _get_emb_model(self, emb_model:str):
        """
        Loads and returns a SentenceTransformer embedding model based on the provided model name.

        Parameters:
            emb_model (str): The name of the embedding model to load. 
                Supported values include:
                    - 'ember-v1'
                    - 'SFR-Embedding-2_R'
                    - 'gte-Qwen2-7B-instruct'
                    - 'phobert'

        Returns:
            SentenceTransformer or PhoBERTWrapper: The loaded embedding model instance.
        """
        if emb_model == 'ember-v1':
            return SentenceTransformer('llmrails/ember-v1', device=self.device)
        
        elif emb_model == 'SFR-Embedding-2_R':
            return SentenceTransformer(f'Salesforce/{emb_model}', device=self.device)
        
        elif emb_model == 'gte-Qwen2-7B-instruct':
            model = SentenceTransformer(f"Alibaba-NLP/gte-Qwen2-7B-instruct", trust_remote_code=True)
            # # In case you want to reduce the max seq length
            # model.max_seq_length = 8192
            return model
        
        elif emb_model in ['phobert', 'vinai/phobert-base']:
            return PhoBERTWrapper('vinai/phobert-base', self.device)
        
        elif emb_model in ['bert', 'google-bert/bert-base-uncased']:
            # Note: Raw BERT embeddings might not be optimal for semantic similarity compared to SBERT models
            return SentenceTransformer('google-bert/bert-base-uncased', device=self.device)
            
        else:
            print(f"Warning: Unknown embedding model '{emb_model}'. Using default multilingual model.")
            return SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2', device=self.device)
        
    def _process_kwds(self, kwd:str):
        """
        Preprocesses a keyword string by lowercasing, removing punctuation, and lemmatizing each word.

        Parameters:
            kwd (str): The keyword or phrase to process.

        Returns:
            tuple: A tuple containing:
                - post_kwd (str): The processed keyword string.
                - len_kwd (int): The number of words in the processed keyword.
        """
        post_kwd = kwd.lower()
        # remove any punctuations as it is not useful for embeddings
        post_kwd = re.sub(r'[^\w\s]', '', post_kwd)
        # Uncomment below to filter based on stop words
        # post_kwd = ' '.join([self.lemmatizer.lemmatize(word, self._get_pos_tag(word)) for word in post_kwd.split() if word not in self.stopwords])
        post_kwd = ' '.join([self.lemmatizer.lemmatize(word, self._get_pos_tag(word)) for word in post_kwd.split()])
        #TODO - convert a digit to string data using 'inflect' module
        # If after postprocessing the string is empty use the keyword string as it is.
        if len(post_kwd)==0: post_kwd = kwd.lower()
        len_kwd = len(post_kwd.split())

        return (post_kwd, len_kwd)

    def _get_pos_tag(self, word:str):
        """
        Maps a word to its respective Part-of-Speech (POS) tag for lemmatization.

        Parameters:
            word (str): The word to tag.

        Returns:
            str: The WordNet POS tag (ADJ, NOUN, VERB, or ADV). Defaults to NOUN if not found.
        """
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J":wordnet.ADJ, "N":wordnet.NOUN, "V":wordnet.VERB, "R":wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    
    def _get_kwd_score(self, ans_embs, len_ans, preds, pred_kwd_embs):
        """
        Computes the maximum cosine similarity between answer embeddings and n-grams from predictions.

        For each prediction-answer pair, generates n-grams (n = len(answer)), computes cosine similarities in bulk,
        and returns the maximum similarity along with the n-gram that produced it.

        Parameters:
            ans_embs (np.ndarray): Embeddings for the answers.
            len_ans (np.ndarray): Lengths of the processed answers.
            preds (list): List of prediction strings.
            pred_kwd_embs (list or np.ndarray): Precomputed prediction keyword embeddings or empty list.

        Returns:
            tuple: (result, ngram_embs)
                - result (list): List of [max_similarity, ngram] for each prediction.
                - ngram_embs (np.ndarray): Embeddings for all n-grams.
        """
        all_ngrams, ans_indices, ngram_counts = [],[],[]

        # Iterate over each prediction-answer pair and generate ngram values
        for i, (pred, ans_len) in enumerate(zip(preds, len_ans)):
            pred_tokens = pred.split()
            n = ans_len
            if len(pred_tokens)<n:
                ngrams = [pred]
            else:
                ngrams = [' '.join(pred_tokens[j:j+n]) for j in range(len(pred_tokens)-n+1)]
            all_ngrams.extend(ngrams) # flattens all ngrams
            ans_indices.extend([i]*len(ngrams))
            ngram_counts.append(len(ngrams))

        # Generate embeddings for all the ngrams, but if 'pred_kwd_embs' are precomputed, just load the embeddings
        ngram_embs = self.emb_model.encode(all_ngrams, batch_size=32, show_progress_bar=False, normalize_embeddings=False, convert_to_numpy=True) if isinstance(pred_kwd_embs, list) else pred_kwd_embs

        # Repeat the answer embeddings
        ans_rep = ans_embs[np.array(ans_indices)]
        cos_matrix = cosine_similarity(ngram_embs, ans_rep)
        cos_sims = np.diag(cos_matrix)

        # Extract the best embedding and it's corresponding ngram
        offsets = np.cumsum([0] + ngram_counts)
        result = []

        for i in range(len(preds)):
            start, end = offsets[i], offsets[i+1]
            sub_cos = cos_sims[start:end]
            if len(sub_cos) == 0:
                print(f'_get_kwd_score_v2 -> missing cosine sim values for pred: {preds[i]}')
                result.append([0.0, ""])
            else:
                max_idx = int(np.argmax(sub_cos))
                result.append([sub_cos[max_idx], all_ngrams[start+max_idx]])

        return result, ngram_embs
        
    def _generate_emb_scores(self, qa_set):
        """
        Generates embedding-based similarity scores (on sentence & token level) for a QA set.

        Parameters:
            qa_set (np.ndarray): Array containing question-answer-prediction sets.

        Returns:
            dict: Dictionary with sentence embedding scores, keyword embedding scores, and max similarity words.
        """
        # Flatten the 'ans' and 'preds'
        string_ans_preds = [item for sublist in qa_set[:, 2:] for item in sublist]
        # Perform preprocessing and prepares for 'keyword embedding score'
        # actual answers are present at index '1'
        # predictions are present at index '3'
        proc_ans, len_ans = self.vect_process_kwds(qa_set[:,1])

        if not self.load_emb_folder:
            if self.verbose: print(" > Generating sentence embeddings...")
            embs = self.emb_model.encode(string_ans_preds, batch_size=32, device=self.device, show_progress_bar=True if self.verbose else False)
            # Extract the synthetic answers and pred embeddigns
            ans_embs = embs[::2]
            pred_embs = embs[1::2]

            # Generates embeddings for 'actual ground-truth'
            kwd_ans_embs = self.emb_model.encode(proc_ans.tolist(), batch_size=32, device=self.device, show_progress_bar=True if self.verbose else False)
            if self.save_emb_folder:
                # Save syn_ans_emb scores, pred_sent_emb scores & ans_kwd_emb
                if self.verbose: print(f'  > Saving synthetic answer embeddings at -> {self.SAVE_EMB_PATHS["syn_ans_emb"]}')
                np.save(self.SAVE_EMB_PATHS["syn_ans_emb"], ans_embs)
                
                if self.verbose: print(f'  > Saving prediction sentence embeddings at -> {self.SAVE_EMB_PATHS["pred_sent_emb"]}')
                np.save(self.SAVE_EMB_PATHS["pred_sent_emb"], pred_embs)
                
                if self.verbose: print(f'  > Saving answer keyword embeddings at -> {self.SAVE_EMB_PATHS["ans_kwd_emb"]}')
                np.save(self.SAVE_EMB_PATHS["ans_kwd_emb"], kwd_ans_embs)

            pred_kwd_embs = []
        else:
            try:
                # Loads the precomputed embeddings
                if self.verbose: print(" > Loading syn_ans_embs & ans_kwd_embs")
                if os.path.exists(self.SAVE_EMB_PATHS['syn_ans_emb']):
                    ans_embs = np.load(self.SAVE_EMB_PATHS["syn_ans_emb"])
                else:
                    if self.verbose: print("  > There are no precomputed embeddings for 'syn ans'")
                    # If it is a new set of synthetic answers from new model
                    # Synthetic answers are present at index '2'
                    ans_embs = self.emb_model.encode(qa_set[:,2], batch_size=32, device=self.device, show_progress_bar=True if self.verbose else False)
                    if self.verbose: print(f'  > Saving synthetic answer embeddings at -> {self.SAVE_EMB_PATHS["syn_ans_emb"]}')
                    np.save(self.SAVE_EMB_PATHS["syn_ans_emb"], ans_embs)

                # This will always be present if save_emb_folder config is run once.
                kwd_ans_embs = np.load(self.SAVE_EMB_PATHS["ans_kwd_emb"])

                # If we need to load the embeddings
                if not os.listdir(self.load_emb_folder):
                    if self.verbose: print("  > There are no precomputed embeddings for prediction set, generating embeddings and saving the embeddings")
                    # Generate the embeddings and save it
                    # Last index contains all the predictions
                    pred_embs = self.emb_model.encode(qa_set[:,-1], batch_size=32, device=self.device, show_progress_bar=True if self.verbose else False)
                    if self.verbose: print(f'   > Saving prediction sentence embeddings at -> {self.SAVE_EMB_PATHS["pred_sent_emb"]}')
                    np.save(self.SAVE_EMB_PATHS["pred_sent_emb"], pred_embs)
                    pred_kwd_embs = []

                else:
                    if self.verbose: print("  > Loading pred_sent_embs & pred_kwd_embs")
                    pred_embs = np.load(self.SAVE_EMB_PATHS["pred_sent_emb"])
                    pred_kwd_embs = np.load(self.SAVE_EMB_PATHS["pred_kwd_emb"])

            except Exception as e:
                if self.verbose: print(f'_generate_emb_scores(), embs/ kwd_ans_embs : {e}')
                sys.exit()
        
        if self.verbose: print(' > Generating sent_emb_scores...')
        sent_emb_scores = np.diag(cosine_similarity(ans_embs, pred_embs))

        # Performs preprocessing on model predictions
        proc_preds, _ = self.vect_process_kwds(qa_set[:,3])
        kwd_scores = [] # contains [(kwd_score, word i.e. matched),....]
        save_kwd_embs = []
        if self.verbose: print(' > Generating kwd_emb_scores...')

        # Generating kwd_emb_scores & max_sim_words
        kwd_scores, save_kwd_embs = self._get_kwd_score(kwd_ans_embs, len_ans, proc_preds, pred_kwd_embs)

        # save the pred_kwd_emb
        if self.save_emb_folder or (self.load_emb_folder and not os.listdir(self.load_emb_folder)):
            if self.verbose: print(f' > Saving pred_kwd_emb at -> {self.SAVE_EMB_PATHS["pred_kwd_emb"]}')
            np.save(self.SAVE_EMB_PATHS["pred_kwd_emb"], np.array(save_kwd_embs))
        
        return {
            'sent_emb_scores': sent_emb_scores,
            'kwd_emb_scores': np.array([score for score, _ in kwd_scores]),
            'max_sim_words': [words for _, words in kwd_scores]
        }

    def _get_bins(self, eval_scores):
        """
        Assigns evaluation scores to bins for discretization.

        Parameters:
            eval_scores (np.ndarray): Array of evaluation scores (float values between 0 and 1).

        Returns:
            np.ndarray: Array of bin indices (integers).
        """
        # Define min and max values
        min_val = 0
        max_val = 1

        # Calculate bin edges (6 bins means we need 7 edges)
        bin_edges = np.linspace(min_val, max_val, 7)

        # Assign each value to a bin
        bin_indices = np.digitize(eval_scores, bin_edges) - 1

        # Ensure the values are within the range - (0,5)
        bin_indices = np.clip(bin_indices, 0, 5)

        return bin_indices

    def _eval_exact_match(self, ans, preds):
        """
        Calculates the fraction of exact keyword matches between answers and predictions.

        Parameters:
            ans (list or np.ndarray): List of answer strings.
            preds (list or np.ndarray): List of prediction strings.

        Returns:
            np.ndarray: Array of fraction scores for each answer-prediction pair.
        """
        # Perform some initial pre-processing of the text
        proc_ans, len_ans = self.vect_process_kwds(ans)
        proc_preds, len_preds = self.vect_process_kwds(preds)

        # Check for the fraction matches
        fraction_scores = []
        for i, (a, p, l_a) in enumerate(zip(proc_ans, proc_preds, len_ans)):
            a_set, p_set = set(a.split()), set(p.split())
            frac_match = len(a_set & p_set) / l_a
            fraction_scores.append(frac_match)

        return np.array(fraction_scores)

    def generate_scores(self, qa_set:list)->dict:
        """
        Generates SMILE evaluation scores for a QA set.

        Parameters:
            qa_set (np.ndarray): Array containing (question, ans, syn_ans/ans, pred) for each sample.

        Returns:
            dict: Dictionary containing all computed scores and metrics.
        """
        results = self._generate_emb_scores(qa_set)
        sent_scores, kwd_emb_scores = results['sent_emb_scores'], results['kwd_emb_scores']

        # Compute keyword score, either using exact matching or directly using the keyword embedding score
        if self.use_exact_matching:
            results['frac_exact_match'] = self._eval_exact_match(qa_set[:,1], qa_set[:,3])
            frac_exact_matches = results['frac_exact_match']
            kwd_scores = (kwd_emb_scores + frac_exact_matches)/2
        else:
            kwd_scores = kwd_emb_scores

        results['kwd_scores'] = kwd_scores
        # Aggregate final SMILE score depending upon sentence/token level embedding scores
        for metric in self.eval_metrics:
            if metric == 'avg':
                results['avg'] = (sent_scores + kwd_scores)/2
            elif metric == 'wt avg':
                results['wt avg'] = (self.avg_w1*sent_scores) + (self.avg_w2*kwd_scores)
            elif metric == 'hm':
                results['hm'] = 2*sent_scores*kwd_scores / (sent_scores+kwd_scores)
            elif metric == 'wt hm':
                results['wt hm'] = 1/((self.hm_w1/sent_scores) + (self.hm_w2/kwd_scores))
            # Assign bins
            if self.assign_bins:
                results[metric+' score'] = self._get_bins(results[metric])
        
        
        # Empty the memory used by embedding model
        torch.cuda.empty_cache()
        return results