import math
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

from baseline_code.baseline_numeric_fewshot import load_math
from unsupervised_icl.manyshot import create_and_return_manyshot_prompt_as_str as create_unsupervised_prompt, SOLVE_PROMPT as unsupervised_prompt_template
from reinforced_icl.manyshot_extracted_examples import create_and_return_manyshot_prompt_as_str as create_reinforced_prompt
from reinforced_icl.manyshot_extracted_examples import load_math_reinforced, SOLVE_PROMPT as reinforced_prompt_template
from supervised_icl.manyshot import create_and_return_manyshot_prompt_as_str as create_supervised_prompt, SOLVE_PROMPT as supervised_prompt_template, load_math500


# print(supervised_prompt_template)
# print(reinforced_prompt_template)
# print(unsupervised_prompt_template)

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
model.eval()

# llm = LLM(
#         model="meta-llama/Llama-3.1-8B-Instruct",
#         download_dir="/home/rveerara/.cache/huggingface/hub",
#         max_model_len=80000,
#         gpu_memory_utilization=0.95,
#     )
# sampling_params = SamplingParams(
#     temperature=0,
#     max_tokens=0,
#     stop_token_ids=[128001, 128008, 128009],
#     logprobs=True,
#     logits_processors=None
# )

def compute_perplexity(text: str) -> tuple[float, int]:
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        
    prompt_token_count = len(inputs["input_ids"][0])

    if prompt_token_count <= 1:
        perplexity = float('inf')
    else:
        avg_neg_log_likelihood = outputs.loss.item()
        perplexity = math.exp(avg_neg_log_likelihood)
        
    return perplexity, prompt_token_count

dataset_math_train = load_math()
all_question_shots_from_math_train = [instance['question'] for instance in dataset_math_train]
all_question_answer_shots_from_math_train = [
        (instance["question"], instance["reasoning"], instance["answer"])
        for instance in dataset_math_train
    ]

all_question_answer_shot_synthetic = load_math_reinforced()
all_question_answer__shots_from_synthetic = [
        (instance["question"], instance["reasoning"], instance["answer"])
        for instance in all_question_answer_shot_synthetic
    ]
print(all_question_answer__shots_from_synthetic[0])

math500_questions = load_math500()

num_shots_list = [4, 5, 10, 25, 50, 75, 100, 125]
supervised_ppls = []
unsupervised_ppls = []
reinforced_ppls = []
supervised_tokens_list = []
unsupervised_tokens_list = []
reinforced_tokens_list = []

for num_shots in num_shots_list:
    supervised_prompt = create_supervised_prompt(all_question_answer_shots_from_math_train, num_shots=num_shots)
    unsupervised_prompt = create_unsupervised_prompt(all_question_shots_from_math_train, num_shots=num_shots)
    reinforced_prompt = create_reinforced_prompt(all_question_answer__shots_from_synthetic, num_shots=num_shots)


    # print(reinforced_prompt)
    
    supervised_prompt = supervised_prompt_template.format(supervised_prompt, math500_questions[0]["question"])
    
    unsupervised_prompt = unsupervised_prompt_template.format(unsupervised_prompt, math500_questions[0]["question"])
    
    reinforced_prompt = reinforced_prompt_template.format(reinforced_prompt, math500_questions[0]["question"])
    
    # print(reinforced_prompt)
    # exit(0)
    
    supervised_ppl, supervised_tokens = compute_perplexity(supervised_prompt)
    unsupervised_ppl, unsupervised_tokens = compute_perplexity(unsupervised_prompt)
    reinforced_ppl, reinforced_tokens = compute_perplexity(reinforced_prompt)
    
    supervised_ppls.append(supervised_ppl)
    unsupervised_ppls.append(unsupervised_ppl)
    reinforced_ppls.append(reinforced_ppl)
    supervised_tokens_list.append(supervised_tokens)
    unsupervised_tokens_list.append(unsupervised_tokens)
    reinforced_tokens_list.append(reinforced_tokens)
    
    print(f"Num shots: {num_shots}")
    print(f"Supervised Perplexity: {supervised_ppl} ({supervised_tokens} tokens)")
    print(f"Unsupervised Perplexity: {unsupervised_ppl} ({unsupervised_tokens} tokens)")
    print(f"Reinforced Perplexity: {reinforced_ppl} ({reinforced_tokens} tokens)")
    print()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(num_shots_list, supervised_ppls, marker='o', label='Supervised')
plt.plot(num_shots_list, unsupervised_ppls, marker='s', label='Unsupervised')
plt.plot(num_shots_list, reinforced_ppls, marker='^', label='Reinforced')
plt.xlabel('Number of Shots')
plt.ylabel('Perplexity')
plt.title('Perplexity vs Number of Shots')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(num_shots_list, supervised_tokens_list, marker='o', label='Supervised')
plt.plot(num_shots_list, unsupervised_tokens_list, marker='s', label='Unsupervised')
plt.plot(num_shots_list, reinforced_tokens_list, marker='^', label='Reinforced')
plt.xlabel('Number of Shots')
plt.ylabel('Number of Tokens')
plt.title('Tokens vs Number of Shots')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('perplexity_and_tokens_plot.png')
plt.close()