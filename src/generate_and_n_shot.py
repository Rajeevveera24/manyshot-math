import json
import os
import re

from tqdm.auto import tqdm

from vllm import LLM, SamplingParams

def load_math(path="../datasets/MATH", split="train"):
    with open(os.path.join(path, split, "dataset.json")) as f:
        data = json.load(f)
    
    examples = [{
        'question': q,
        'answer': a,
    } for q, a in zip(data['question'], data['extracted_answers'])]

    return examples

data = load_math(split='test')
len(data)

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct", 
    download_dir="/home/amittur/.cache/huggingface/hub", 
    max_model_len=20000,
    gpu_memory_utilization=0.99,
    max_num_seqs=50,
)


def get_n_shot_cot_solve_prompts(data, synth_data):
    SOLVE_N_SHOT_PROMPT = """Some examples of math problems and their answers, similar to the one you are going to be asked are provided below. Use them to understand and solve the problem. End your response with "<|eot_id|>".    

### Examples ###
{examples}

### Output Format ###
Question: <the math problem you need to solve> 
Reasoning: <your step by step reasoning for the answer>
Answer: <your final answer only>

### Input ###
Question: {question}
"""
    return [
        SOLVE_N_SHOT_PROMPT.format(
            examples="\n\n".join([f"Question: {d['question']}\nAnswer: {d['answer']}" for d in synth_data[i]]), 
            question=d['question']
        ) 
        for i, d in enumerate(data)
    ]


def get_generate_prompt(data, n_samples=1):
    GENERATE_PROMPT = """Given a reference math problem, generate {n_samples} similar math problems, along with their answers. End your response with "<|eot_id|>".
Use this format for each generated problem:
Question: <new math problem>
Answer: <answer to the math problem>

Question: {question}
"""

    return [GENERATE_PROMPT.format(n_samples=n_samples, question=d['question']) for d in data]

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=5120,
    stop_token_ids=[128001, 128008, 128009],
)

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
    matches = re.findall(r"Question: (.*?)\nAnswer: (.*?)(?=\s*Question:|$)", text, re.DOTALL)
    return [{
        'question': q.strip('\n '),
        'answer': a.strip('\n ')
    } for q, a in matches]


# generate_prompts = get_generate_prompt(data, 3)

# BATCH_SIZE = 64

# synth_data = []

# for i in tqdm(range(0, len(generate_prompts), BATCH_SIZE)):
#     outputs = llm.generate(generate_prompts[i:i+BATCH_SIZE], sampling_params)
#     for output in outputs:
#         generated_text = output.outputs[0].text
#         synth_data.append(extract_synthetic_data(generated_text))

# with open("synth_data.json", "w") as f:
#     json.dump(synth_data, f)

with open("synth_data.json", "r") as f:
    synth_data = json.load(f)

solve_n_shot_prompts = get_n_shot_cot_solve_prompts(data, synth_data)

answers = []

outputs = llm.generate(solve_n_shot_prompts, sampling_params)
for output in outputs:
    generated_text = output.outputs[0].text
    answers.append(extract_cot_answer(generated_text))

# Save answers as json
with open("answers_3_shot_synth_64.json", "w") as f:
    json.dump(answers, f)



