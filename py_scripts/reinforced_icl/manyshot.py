import json
import os
from typing import List, Tuple
import dspy
import sys
from dotenv import load_dotenv
from vllm import LLM, SamplingParams
import re
import random
import argparse

# sys.path.append("..") #Adjust as needed
from experiment_logger import ExperimentLogger
from baseline_code.baseline_numeric_fewshot import load_math, extract_answer_numeric
# sys.path.remove("..")

# load_dotenv("../keys.env")

MAX_REINFORCED_EXAMPLES = 4

SOLVE_PROMPT = """
Answer the math problem in the format shown below. End your response with "<|eot_id|>".

The final answer in number format. Ignore any latex that exists in the question, and determine the numerical answer step by step. Reason logically internally and arrive at a final answer. If you get an expression, simplify it and return a numerical answer. If your answer is a fraction, simplify it and return a decimal rounded to 3 places. You will also be shown many example problems to help you get familiarized with the type of the problem that you would need to solve. Do not solve these sample problem 

Your final answer should be a number in the last line. Follow the format below:

--------------------------------------------------
Problem: <you will be given a math question> 
Reasoning: <your step by step reasoning for the answer>
Answer: <your final answer as a number only>
--------------------------------------------------

Here is a list of problems with answers to get you familiar with the types of questions. Do not solve them.

{}

--------------------------------------------------

Here is the problem that you need to solve

Problem: {}
"""

def load_math500(path="../datasets/MATH500", split=None):
    with open(os.path.join(path, "dataset_numeric.json")) as f:
        data = json.load(f)
    
    examples = [{
        "question": q,
        # "reasoning": reasoning, 
        "answer": a,
        "id": id,
    } for q, a, id in zip(data['question'], data['extracted_answers'], data["id"])]

    return examples


def load_reinforce(reinforce_filename, num_questions, num_examples, total_examples):
    with open(reinforce_filename, "r") as f:
        data = json.load(f)
    
    # print(data)
    
    selected_data = []

    for data_instance in data:
        if len(data_instance) >= total_examples:
            selected_data.append(data_instance)
    
    all_selected_examples = []

    for i in range(num_questions):
        # print(len(selected_data), len(selected_data[i]))
        indices = random.sample(list(range(total_examples)), num_examples)
        selected_examples = [selected_data[i][k] for k in indices]
        all_selected_examples.append(selected_examples)
    
    return all_selected_examples


def create_and_return_manyshot_prompt_as_str(example_list: List[Tuple]) -> str:
    example_strs = []
    for examples in example_list:
        example_str = ""
        for example in examples:
            example_str += "\n--------------------------------------------------\n"
            example_str += "\nProblem: "+ example["problem"].rstrip("\n") + "\n\nReasoning: " + example["reasoning"].rstrip("\n") + "\n\nAnswer: " + example["answer"].rstrip("\n") + "\n\n"
        example_strs.append(example_str)
    return example_strs

def build_prompts(dataset_math_test, example_strs):
    prompts = []
    for d, e in zip(dataset_math_test, example_strs):
        prompt = SOLVE_PROMPT.format(e, d['question'])
        prompts.append(prompt)
    return prompts
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_shots', type=int, default=2,
                       help='Number of shots/examples to use')
    args = parser.parse_args()
    num_shots = args.num_shots
    
    dataset_math_test = load_math500(split="test")
    dataset_math_train = load_reinforce("../src/synth_data/cot_synth_data_70b_int4_50.json", len(dataset_math_test), num_shots, num_shots)
    example_strs = create_and_return_manyshot_prompt_as_str(dataset_math_train)
    final_input_prompts = build_prompts(dataset_math_test, example_strs)
    
    # print(final_input_prompts[0])
    # exit(0) 
    
    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct", 
        download_dir="/home/rveerara/.cache/huggingface/hub", 
        max_model_len=80000,
        gpu_memory_utilization=0.95
    )
    
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=5120,
        stop_token_ids=[128001, 128008, 128009],
    )
    
    batch_size = 100
    logger = ExperimentLogger(results_file="../experiments/rveerara/"+str(num_shots)+"shot_reinforced.json", logging_frequency=batch_size, use_existing=False)
    correct_count = 0
    incorrect_count = 0 
    failed_count = 0

    for batch_start in range(0, len(final_input_prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(final_input_prompts))
        batch_prompts = final_input_prompts[batch_start:batch_end]
                
        outputs = llm.generate(batch_prompts, sampling_params)

        batch_results = []
        for i, output in enumerate(outputs):
            idx = batch_start + i
            generated_text = output.outputs[0].text
            extracted, parsed = extract_answer_numeric(generated_text)
            
            
            batch_results.append({
                "question_id": dataset_math_test[idx]["id"],
                "predicted_answer": extracted["answer"]
            })
            
            if not parsed:
                failed_count += 1
                continue
            
            try:
                gt_answer = float(dataset_math_test[idx]["answer"])
            except Exception as e:
                incorrect_count += 1
                continue
                
            if extracted["answer"] == gt_answer:
                correct_count += 1
            else:
                incorrect_count += 1
        
        total = correct_count + incorrect_count + failed_count
        print(f"Progress: {total}/{len(final_input_prompts)} questions processed")
        print(f"Correct: {correct_count}, Incorrect: {incorrect_count}, Unparsed: {failed_count}")
        print(f"Accuracy: {correct_count/total:.2%}")
        logger.log_results(batch_results)
        logger.write_results_to_file()
