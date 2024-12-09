import argparse
import json
import os
import re
import sys

from tqdm.auto import tqdm

from vllm import LLM, SamplingParams

# Add argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--n_samples', type=int, default=3, help='Number of samples to generate per problem')
parser.add_argument('--generate', action='store_true', help='Generate synthetic data')
parser.add_argument('--generate_cot', action='store_true', help='Generate CoT synthetic data if set')
parser.add_argument('--predict', action='store_true', help='Predict answers')
parser.add_argument('--pred_use_cot', action='store_true', help='Use CoT synthetic data for prediction')
parser.add_argument('--supervised', action='store_true', help='Use supervised data for prediction')
parser.add_argument('--math500', action='store_true', help='Use MATH500 dataset')
args = parser.parse_args()

print(args)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
# os.environ["NCCL_P2P_DISABLE"] = "1"

def load_math(path="../datasets/MATH", split="train"):
    with open(os.path.join(path, split, "dataset.json")) as f:
        data = json.load(f)
    
    examples = [{
        'question': q,
        'answer': a,
    } for q, a in zip(data['question'], data['extracted_answers'])]

    return examples

math500_idxs = None
if args.math500:
    data = load_math(path='../datasets/MATH500', split='')
    with open('../datasets/MATH500inMATH-idxs.json') as f:
        math500_idxs = json.load(f)
else:
    data = load_math(split='test')
len(data)

if args.predict:
    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct", 
        download_dir="/home/amittur/.cache/huggingface/hub", 
        max_model_len=30000,
        tensor_parallel_size=1,
        # gpu_memory_utilization=0.99,
        # max_num_seqs=50,
    )
elif args.generate:
    llm = LLM(
        model="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        tensor_parallel_size=2,
        # download_dir="/home/amittur/.cache/huggingface/hub", 
        max_model_len=50000,
        # gpu_memory_utilization=0.95
        # max_num_seqs=128,
    )

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=5120,
    stop_token_ids=[128001, 128008, 128009],
)


def get_solve_prompts(data):
    SOLVE_PROMPT = """Answer the math problem in the format shown below. End your response with "<|eot_id|>".

---
Problem: <you will be given a math problem> 
Reasoning: <your step by step reasoning for the answer>
Answer: <your final answer only>
---

Problem: {}
"""

    return [SOLVE_PROMPT.format(d['question']) for d in data]


def get_n_shot_cot_solve_prompts(data, synth_data):
    if args.supervised:
        SOLVE_N_SHOT_PROMPT = """Some examples of math problems and their answers, similar to the one you are going to be asked are provided below. Use them to understand and solve the problem. End your response with "<|eot_id|>"."""
    else:
        SOLVE_N_SHOT_PROMPT = """Some examples of math problems, similar to the one you are going to be asked are provided below. Use them to understand and solve the problem. End your response with "<|eot_id|>"."""    

    SOLVE_N_SHOT_PROMPT += """

### Examples ###
{examples}

### Output Format ###
Problem: <the math problem you need to solve> 
Reasoning: <your step by step reasoning for the answer>
Answer: <your final answer only>

### Input ###
Problem: {problem}
"""

    if args.supervised:
        examples = ["\n\n".join([f"Problem: {d['problem']}\nAnswer: {d['answer']}" for d in sd]) for sd in synth_data]
    else:
        examples = ["\n".join([f"Problem: {d['problem']}" for d in sd]) for sd in synth_data]

    return [
        SOLVE_N_SHOT_PROMPT.format(
            examples=examples[i],
            problem=d['question']
        ) 
        for i, d in enumerate(data)
    ]


def get_generate_prompt(data, n_samples=1):
    GENERATE_PROMPT = """Given a reference math problem, generate {n_samples} similar math problems, along with their answers. End your response with "<|eot_id|>".
Use this format for each generated problem:
Problem: <new math problem>
Answer: <answer to the math problem>

Problem: {problem}
"""

    return [GENERATE_PROMPT.format(n_samples=n_samples, problem=d['question']) for d in data]

def get_generate_prompt_cot(data, n_samples=1):
    GENERATE_PROMPT = """Given a reference math problem, generate {n_samples} similar math problems, along with their answers and reasoning for the answer. End your response with "<|eot_id|>".
Use this format for each generated problem:
Problem: <new math problem>
Reasoning: <step by step reasoning for the answer>
Answer: <answer to the math problem>

Problem: {problem}
"""

    return [GENERATE_PROMPT.format(n_samples=n_samples, problem=d['question']) for d in data]


def extract_cot_answer(text):
    match = re.search(r"Reasoning: (.+)Answer: (.+)", text, re.DOTALL)
    answer = reasoning = ''
    if match:
        reasoning = match.group(1).strip('\n ')
        answer = match.group(2).strip('\n ')
    return {
        'answer': answer,
        'reasoning': reasoning
    }


def extract_synthetic_data(text):
    matches = re.findall(r"Problem: (.*?)\nAnswer: (.*?)(?=\s*Problem:|$)", text, re.DOTALL)
    return [{
        'problem': q.strip('\n '),
        'answer': a.strip('\n ')
    } for q, a in matches]

def extract_synthetic_data_cot(text):
    matches = re.findall(r"Problem: (.*?)\nReasoning: (.*?)\nAnswer: (.*?)(?=\s*Problem:|$)", text, re.DOTALL)
    return [{
        'problem': q.strip('\n '),
        'answer': a.strip('\n '),
        'reasoning': r.strip('\n ')
    } for q, r, a in matches]


if args.generate:
    if args.generate_cot:
        generate_prompts = get_generate_prompt_cot(data, args.n_samples)
    else:
        generate_prompts = get_generate_prompt(data, args.n_samples)

    synth_data = []

    outputs = llm.generate(generate_prompts, sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text
        if args.generate_cot:
            synth_data.append(extract_synthetic_data_cot(generated_text))
        else:
            synth_data.append(extract_synthetic_data(generated_text))

    # Update the output path to use n_samples
    # synth_data_path = f"synth_data/{'cot_' if args.generate_cot else ''}synth_data_70b_int4_{args.n_samples}.json"
    synth_data_path = f"synth_data/math500/{'cot_' if args.generate_cot else ''}synth_data_70b_int4_{args.n_samples}.json"
    print(f"Generate Synth Data Path: {synth_data_path}")
    with open(synth_data_path, "w") as f:
        json.dump(synth_data, f)

if args.predict:
    # Update the input path to use n_samples
    synth_data_path = f"synth_data/{'cot_' if args.pred_use_cot else ''}synth_data_70b_int4_{args.n_samples}.json"
    print(f"Predict Synth Data Path: {synth_data_path}")
    with open(synth_data_path, "r") as f:
        print(f.name)
        synth_data = json.load(f)
    
    if args.math500:
        synth_data = [synth_data[i] for i in math500_idxs]

    solve_n_shot_prompts = get_n_shot_cot_solve_prompts(data, synth_data)

    answers = []

    outputs = llm.generate(solve_n_shot_prompts, sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text
        answers.append(extract_cot_answer(generated_text))

    # Save answers as json
    answers_path = f"{'un' if not args.supervised else ''}supervised_icl/{'cot_' if args.pred_use_cot else ''}answers_{args.n_samples}_shot_synth_8b.json"
    print(f"Predict Answers Path: {answers_path}")
    with open(answers_path, "w") as f:
        json.dump(answers, f)



