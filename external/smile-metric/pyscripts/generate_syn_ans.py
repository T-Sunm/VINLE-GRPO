import openai
import concurrent.futures
import argparse
import json
import os
import time
from tqdm import tqdm
import sys
import requests
import utils
import torch
from transformers import pipeline
from huggingface_hub import login
import warnings
warnings.simplefilter('ignore')

openai.api_key = '<openai_api_key>'

def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Add all the arguments
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output file.')
    # Supports openai models, llama & deepseek-r1-600b model
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', required=True, help='Model to be used for synthetic answer generation. Note: Make sure the model name is the EXACT MODEL NAME as on huggingface.')

    args = parser.parse_args()
    return args

def get_deepseek_response(data, max_tries=3, delay_seconds=5):
    """
    Uses DeepSeek models to generate a synthetic answer from a question-answer pair.

    Parameters:
        data (dict): Dictionary containing 'question' and 'answer'.
        max_tries (int): Maximum number of retry attempts for the API call.
        delay_seconds (int): Delay between retries in seconds.

    Returns:
        str: The generated synthetic answer as a single sentence.
    """
    # TODO - add support for when 'answer' is a list.
    # Define the URL
    url = "<add deepseek url>"

    # Define the API key
    api_key = "<add deepseek api key>"

    # Define the request headers
    headers = {
        # "Authorization": f"Bearer {api_key}",
        "X-Api-Key" : f"{api_key}",
        "Content-Type": "application/json"
    }

    # Define the request payload (modify as needed)
    data = {
    "messages": [
        {
            "role":"system",
            "content":
            "You are an intelligent chatbot designed for generating answer as a sentence from question-answer pairs. "
            "Your task is to generate a single sentence answer using the question and the answer already provided. Here's how you can accomplish the task:"
            "------"
            "##INSTRUCTIONS: "
            "- Look at the provided answer.\n"
            "- Generate a short single sentence response using the question and the answer.\n"
            "- Response SHOULD ALWAYS USE THE WORDS FROM ANSWER provided."
            "- DO NOT USE THE QUESTION AS IT IS IN THE RESPONSE."
            "- Return only the response and nothing else."
        },
        {
            "role": "user",
            "content":
                "Please phrase a short single sentence answer using question-answer pair only:\n\n"
                f"Question: {data['question']}\n"
                f"Answer: {data['answer']}\n"
                "DO NOT PROVIDE ANY OTHER OUTPUT APART FROM A SINGLE SHORT SENTENCE."
                
        }
    ],
    "stream": False
    }

    for attempt in range(max_tries):
        try:
            # Send the POST request
            completion = requests.post(url, json=data, headers=headers)

            # Print response
            if completion.status_code >= 200 and completion.status_code<300:
                json_data = completion.json()
                response_msg = json_data['choices'][0]['delta']['content'].strip()

                #response_msg contains the text in the format - <think>...</think>...response
                response_msg = response_msg.split("</think>")[-1].strip()
                break

            else:
                print(f'deepseek request failed with status code {completion.status_code}')
            
            
        except Exception as e:
            print(f"deepseek error: {e}")

        if attempt < max_tries-1:
            time.sleep(delay_seconds)

    return response_msg

def get_openai_response(model, data):
    """
    Uses OpenAI models to generate a synthetic answer from a question-answer pair.

    Parameters:
        model (str): Name of the OpenAI model to use.
        data (dict): Dictionary containing 'question' and 'answer'.

    Returns:
        str or list: The generated synthetic answer(s) as a single sentence or list of sentences.
    """
    if isinstance(data['answer'], list):
       ans = data['answer']
    elif isinstance(data['answer'], str):
        ans = [data['answer'].strip()]
    
    response_msgs = []
    for a in ans:
        completion = openai.chat.completions.create(
            model = model,
            messages = [
                {
                    "role":"system",
                    "content":
                    "You are an intelligent chatbot designed for generating answer as a sentence from question-answer pairs. "
                    "Your task is to generate a single sentence answer using the question and the answer already provided. Here's how you can accomplish the task:"
                    "------"
                    "##INSTRUCTIONS: "
                    "- Look at the provided answer.\n"
                    "- Generate a short single sentence response using the question and the answer.\n"
                    "- Response SHOULD ALWAYS USE THE WORDS FROM ANSWER provided."
                    "- DO NOT USE THE QUESTION AS IT IS IN THE RESPONSE."
                    "- Return only the response and nothing else."
                },
                {
                    "role": "user",
                    "content":
                        "Please phrase a short single sentence answer using question-answer pair only:\n\n"
                        f"Question: {data['question']}\n"
                        f"Answer: {a}\n"
                        "DO NOT PROVIDE ANY OTHER OUTPUT APART FROM A SINGLE SHORT SENTENCE."
                        
                }
            ]
        )
        response_msgs.append(completion.choices[0].message.content)

    return response_msgs if isinstance(data['answer'], list) else response_msgs[0]

def convert_data_to_prompt(question, answer):
    """
    Converts a question and answer into a prompt format for LLMs.

    Parameters:
        question (str): The question string.
        answer (str): The answer string.

    Returns:
        list: List of message dictionaries formatted for LLM input.
    """
    messages = [
        {
            "role":"system",
            "content":
            "You are an intelligent chatbot designed for generating answer as a sentence from question-answer pairs. "
            "Your task is to generate a single sentence answer using the question and the answer already provided. Here's how you can accomplish the task:"
            "------"
            "##INSTRUCTIONS: "
            "- Look at the provided answer.\n"
            "- Generate a short single sentence response using the question and the answer.\n"
            "- Response SHOULD ALWAYS USE THE WORDS FROM THE ANSWER AS IT IS .\n"
            "- DO NOT USE THE QUESTION AS IT IS IN THE RESPONSE.\n"
            "- Return only the response and nothing else.\n"
        },
        {
            "role": "user",
            "content":
                "Please phrase a short single sentence answer using question-answer pair only:\n\n"
                f"Question: {question}\n"
                f"Answer: {answer.strip()}\n"
                "DO NOT PROVIDE ANY OTHER OUTPUT APART FROM A SINGLE SHORT SENTENCE."
                
        }
    ]

    return messages

def get_llama_responses(pipe, data):
    """
    Uses a HuggingFace pipeline to generate synthetic answers for a batch of prompts.

    Parameters:
        pipe (transformers.Pipeline): HuggingFace pipeline for text generation.
        data (list): List of prompt messages.

    Returns:
        list: List of generated synthetic answers.
    """
    responses = []
    try:
        responses = pipe(data, 
                        batch_size=8,  
                        max_new_tokens=256,
                        pad_token_id=pipe.tokenizer.eos_token_id)
        # Extract the responses
        responses = [output[0]["generated_text"][-1]['content'].strip() for output in responses]
    except Exception as e:
        print(f'Error processing get_llama_response - {e}')

    return responses

def get_llama_response_seq(pipe, data):
    if isinstance(data['answer'], list):
       ans = data['answer']
    elif isinstance(data['answer'], str):
        ans = [data['answer'].strip()]
    
    response_msgs = []
    for a in ans:
        messages = [
                {
                    "role":"system",
                    "content":
                    "You are an intelligent chatbot designed for generating answer as a sentence from question-answer pairs. "
                    "Your task is to generate a single sentence answer using the question and the answer already provided. Here's how you can accomplish the task:"
                    "------"
                    "##INSTRUCTIONS: "
                    "- Look at the provided answer.\n"
                    "- Generate a short single sentence response using the question and the answer.\n"
                    "- Response SHOULD ALWAYS USE THE WORDS FROM THE ANSWER AS IT IS .\n"
                    "- DO NOT USE THE QUESTION AS IT IS IN THE RESPONSE.\n"
                    "- Return only the response and nothing else.\n"
                },
                {
                    "role": "user",
                    "content":
                        "Please phrase a short single sentence answer using question-answer pair only:\n\n"
                        f"Question: {data['question']}\n"
                        f"Answer: {a}\n"
                        "DO NOT PROVIDE ANY OTHER OUTPUT APART FROM A SINGLE SHORT SENTENCE."
                        
                }
            ]

        outputs = pipe(
            messages,
            max_new_tokens=256,
            pad_token_id=pipe.tokenizer.eos_token_id
        )
        response_msgs.append(outputs[0]["generated_text"][-1]['content'].strip())

    return response_msgs if isinstance(data['answer'], list) else response_msgs[0]

def initialise_llama_model(model_id):
    """
    Initializes and returns a HuggingFace pipeline for a Llama model.

    Parameters:
        model_id (str): The model identifier on HuggingFace.

    Returns:
        transformers.Pipeline: Initialized pipeline for text generation.
    """
    login(token="<add your huggingface token id>")
    pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
            )
    pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
    pipe.tokenizer.padding_side = "left"
    return pipe

def get_response(model, data=None, bad_questions=[], pipe=None, prompts=None):
    """
    Generates a synthetic answer using the specified model (OpenAI, DeepSeek, or Llama).

    Parameters:
        model (str): Model name to use for generation.
        data (dict, optional): Input data containing 'question' and 'answer'.
        bad_questions (list): List to append failed examples.
        pipe (transformers.Pipeline, optional): Pipeline for Llama models.
        prompts (list, optional): List of prompts for batch processing.

    Returns:
        str or list: The generated synthetic answer(s).
    """
    response_msg = '' if prompts is None and isinstance(data['answer'], list) else ['']
    try:
        if 'gpt' in model:
            response_msg = get_openai_response(model, data)
        elif 'deepseek' in model:
            response_msg = get_deepseek_response(data)
        elif 'llama' in model:
            ## Uncomment to run llama model sequentially
            # response_msg = get_llama_response_seq(pipe, data)
            # Since pipeline can't print verbose output, we manually do batching to see the progress
            response_msg = []
            batch_size = 16
            for i in tqdm(range(0, len(prompts), batch_size), desc="batch inference"):
                batch_data = prompts[i:i+batch_size]
                response_msg.extend(get_llama_responses(pipe, batch_data))
        else:
            print('Incorrect model')
            bad_questions.append(data)
    except Exception as e:
        print(f"Error getting reponse for qid - {data['question_id' if 'question_id' in data.keys() else 'id']}")
        print(f"{e}")
        bad_questions.append(data)

    if 'llama' not in model: data['syn_ans'] = response_msg
    return response_msg

def main():
    """
    Main entry point for synthetic answer generation.

    Handles argument parsing, input loading, model initialization, synthetic answer generation,
    error handling, and saving of results.

    Steps:
        1. Parses command-line arguments.
        2. Loads input data.
        3. Initializes the specified model.
        4. Generates synthetic answers for each input.
        5. Handles errors and saves failed examples.
        6. Saves the results to the output file.
    """
    # Parse the cmd line arguments
    args = parse_arguments()

    # Open the input file
    file_ext = args.input_file.split('.')[-1]
    if file_ext == 'jsonl':
        with open(args.input_file, 'r') as f:
            inp_data = [json.loads(line) for line in f]
    elif file_ext == 'json':
        inp_data = json.load(open(args.input_file, 'r'))
    else:
        raise('Invalid input file format, has to be either json, jsonl.')

    results, bad_questions = [], []

    # Initialise model if necessary
    pipe=None
    if 'llama' in args.model:
        pipe = initialise_llama_model(args.model)
        
        # Do llama specific processing
        start_time = time.time()
        # Check if the answer is a list or string
        ans_list = isinstance(inp_data[0]['answer'], list)
        prompts = []
        if ans_list:
            for data in inp_data:
                question = data['question']
                for a in data['answer']:
                    prompts.append(convert_data_to_prompt(question, a))
        else:
            prompts = [convert_data_to_prompt(data['question'], data['answer']) for data in inp_data]

        responses = get_response(model=args.model, pipe=pipe, prompts=prompts)
        start_idx = 0
        # Convert response back to appropriate format
        for i, data in enumerate(inp_data):
            if ans_list:
                end_idx = start_idx + len(data['answer'])
                extract_response = responses[start_idx:end_idx]
                data['syn_ans'] = extract_response if isinstance(extract_response, list) else [extract_response]
                start_idx = end_idx
            else:
                data['syn_ans'] = responses[i]

    else:
        start_time = time.time()
        # Execute task
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(get_response, args.model, data, bad_questions, pipe) for data in inp_data]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                results.append(future.result())

    print(f'Completed questions -> {len(results)}/{len(inp_data)}')
    print(f'Failed questions -> {len(bad_questions)}')
    end_time = time.time()
    utils.time_exec(start_time, end_time, 'Synthetic Answer Generation time')
    
    # Save the data
    os.makedirs(f'{os.path.sep}'.join(args.output_file.split(os.path.sep)[:-1]), exist_ok=True)
    if args.output_file[-1]=='l':
        with open(args.output_file, 'w') as f:
            for item in inp_data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
    else:
        with open(args.output_file,'w') as f:
            json.dump(inp_data, f, indent=4)

    bad_fp = f'{os.path.sep}'.join(args.output_file.split(os.path.sep)[:-1])+'/bad_examples.json'
    print(f'Failed examples saved at -> {bad_fp}')
    with open(bad_fp,'w') as f:
        json.dump(bad_questions, f,indent=4)

if __name__=='__main__':
    main()
