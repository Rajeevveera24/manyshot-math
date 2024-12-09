# %%
import json
import os
import re

from vllm import LLM, SamplingParams

os.environ["NCCL_P2P_DISABLE"] = "1"

# %%
def load_math(path="../datasets/MATH", split="train"):
    with open(os.path.join(path, split, "dataset.json")) as f:
        data = json.load(f)
    
    examples = [{
        'question': q,
        'answer': a,
    } for q, a in zip(data['question'], data['extracted_answers'])]

    return examples

# %%
data = load_math(split='test')
# data = load_math(split='train') + load_math(split='test')
len(data)

# %%
llm = LLM(
    # model="meta-llama/Llama-3.1-8B-Instruct", 
    model="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
    tensor_parallel_size=2,
    # download_dir="/home/amittur/.cache/huggingface/hub", 
    max_model_len=10000,
    # gpu_memory_utilization=0.95
    # max_num_seqs=128,
)

# %%
SOLVE_PROMPT = """
Answer the math problem in the format shown below. End your response with "<|eot_id|>".

---
Problem: <you will be given a math question> 
Reasoning: <your step by step reasoning for the answer>
Answer: <your final answer only>
---

Problem: {}
"""

def get_prompts(data):
    return [SOLVE_PROMPT.format(d['question']) for d in data]

# %%
sampling_params = SamplingParams(
    temperature=0,
    max_tokens=5120,
    stop_token_ids=[128001, 128008, 128009],
)

# %%
def extract_answer(text):
    match = re.search(r"Answer: (.+)", text, re.DOTALL)
    answer = ''
    reasoning = '[NO_COT_REASONING]'
    if match:
        # reasoning = match.group(1).strip()
        answer = match.group(1).strip()
    return {
        'answer': answer,
        'reasoning': reasoning
    }

def extract_cot_answer(text):
    match = re.search(r"Reasoning: (.+)Answer: (.+)", text, re.DOTALL)
    answer = reasoning = ''
    if match:
        reasoning = match.group(1).strip()
        answer = match.group(2).strip()
    return {
        'answer': answer,
        'reasoning': reasoning
    }


# %%
data_prompts = get_prompts(data)

# %%
outputs = llm.generate(data_prompts, sampling_params)
answers = []

for i, output in enumerate(outputs):
    # prompt = output.prompt
    generated_text = output.outputs[0].text

    answers.append(extract_cot_answer(generated_text))

    # print(f"Question Idx: {i}, Response: {generated_text!r}")

# %%
# Save answers as json
with open("answers_70b_int4.json", "w") as f:
    json.dump(answers, f)

