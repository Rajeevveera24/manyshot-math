import json
import os
import dspy
import sys
from dotenv import load_dotenv
from vllm import LLM, SamplingParams
import re

sys.path.append("..")
from py_scripts.experiment_logger import ExperimentLogger
sys.path.remove("..")

# load_dotenv("../keys.env")

SOLVE_PROMPT = """
Answer the math problem in the format shown below. End your response with "<|eot_id|>".

The final answer in number format. Ignore any latex that exists in the question, and determine the numerical answer step by step. Reason logically internally and arrive at a final answer. If you get an expression, simplify it and return a numerical answer. If your answer is a fraction, simplify it and return a decimal rounded to 3 places. You will also be shown upto five exanple problems to help you get familiarized with the type of the problem that you would need to solve. Do not solve these sample problem 

Your final answer should be a number in the last line. Follow the format below:

---
Problem: <you will be given a math question> 
Reasoning: <your step by step reasoning for the answer>
Answer: <your final answer as a number only>
---

Problem 1: {}
Problem 2: {}
Problem 3: {}
Problem 4: {}
Problem 5: {}

Here is the problem that you would need to solve:
Problem: {}
"""

def get_prompts(test_data, train_data):
    return [SOLVE_PROMPT.format(d_[0]['question'], d_[1]['question'], d_[2]['question'], d_[3]['question'], d_[4]['question'], d['question']) for d_, d in zip(train_data, test_data)]

def load_math(path="../datasets/MATH", split="train"):
    with open(os.path.join(path, split, "dataset_numeric.json")) as f:
        data = json.load(f)
    
    examples = [{
        'question': q,
        'answer': a,
        'reasoning': reasoning,
        "id": id,
    } for q, a, reasoning, id in zip(data['question'], data['extracted_answers'], data['answer'], data["id"])]

    return examples

def extract_answer(text):
    match = re.search(r"Reasoning: (.+)Answer: (.+)", text, re.DOTALL)
    answer = reasoning = ""
    if match:
        reasoning = match.group(1).strip()
        answer = match.group(2).strip()
    return {
        'answer': answer,
        'reasoning': reasoning
    }
    
def extract_answer_numeric(text: str):
    match = re.search(r"Reasoning: (.+)Answer: (.+)", text, re.DOTALL)
    answer = 'NO_ANSWER'
    reasoning = 'NO_REASONING'
    parse_successful = False
    try:
        match = re.search(r"Reasoning: (.+)Answer: (.+)", text, re.DOTALL)
        if match:
            reasoning = match.group(1).strip()
            answer = match.group(2).strip()
            answer = float(answer)
            parse_successful = True
    except Exception as e:
        print(f"Error extracting numeric answer: {e}. Text answer is {answer}")
    
    return {
        'answer': answer,
        'reasoning': reasoning
    }, parse_successful

if __name__ == "__main__":
    
    dataset_math_test = load_math(split="test")
    
    dataset_math_train = load_math(split="train")
    
    # Randomly select 5 examples from training set for each test question
    import random
    few_shot_examples = []
    for _ in dataset_math_test:
        few_shot_examples.append(random.sample(dataset_math_train, k=5))
    
    print(type(few_shot_examples[0]))
    # exit(0)
    data_prompts = get_prompts(train_data=few_shot_examples, test_data=dataset_math_test)
    
    print(data_prompts[0])
    # exit(0)
    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct", 
        download_dir="/home/rveerara/.cache/huggingface/hub", 
        max_model_len=10000,
        gpu_memory_utilization=0.95
        # max_num_seqs=128,
    )
    
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=5120,
        stop_token_ids=[128001, 128008, 128009],
    )
    
    batch_size = 500
    logger = ExperimentLogger(results_file='../experiments/rveerara/5shot_unsupervised.json', logging_frequency=batch_size)

    # Initialize counters
    correct_count = 0
    incorrect_count = 0 
    failed_count = 0

    for batch_start in range(0, len(data_prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(data_prompts))
        batch_prompts = data_prompts[batch_start:batch_end]
                
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
        print(f"Progress: {total}/{len(data_prompts)} questions processed")
        print(f"Correct: {correct_count}, Incorrect: {incorrect_count}, Unparsed: {failed_count}")
        print(f"Accuracy: {correct_count/total:.2%}")
        logger.log_results(batch_results)
        logger.write_results_to_file()
