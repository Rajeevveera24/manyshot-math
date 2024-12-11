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

from experiment_logger import ExperimentLogger
from baseline_code.baseline_numeric_fewshot import load_math, extract_answer_numeric

# load_dotenv("../keys.env")
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

    examples = [
        {
            "question": q,
            # "reasoning": reasoning,
            "answer": a,
            "id": id,
        }
        for q, a, id in zip(data["question"], data["extracted_answers"], data["id"])
    ]

    return examples


def load_math_reinforced(path="../datasets/MATH500"):
    with open(os.path.join(path, "MATH_reinforced_numeric.json")) as f:
        data = json.load(f)
    
    examples = [{
        'question': q,
        'answer': a,
        'reasoning': reasoning,
    } for q, a, reasoning in zip(data['question'], data['answers'], data['reasonings'])]
    return examples

def create_and_return_manyshot_prompt_as_str(
    question_list: List[Tuple], num_shots=50
) -> str:
    selected_questions = random.sample(question_list, k=num_shots)
    selected_questions = [
        "\nProblem: "
        + question.rstrip("\n")
        + "\n\nReasoning: "
        + reasoning.rstrip("\n")
        + "\n\nAnswer: "
        + answer.rstrip("\n")
        + "\n\n"
        for i, (question, reasoning, answer) in enumerate(selected_questions)
    ]
    return "\n--------------------------------------------------\n".join(
        selected_questions
    ).rstrip("\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_shots", type=int, default=50, help="Number of shots/examples to use"
    )
    args = parser.parse_args()
    num_shots = args.num_shots

    dataset_math_test = load_math500(split="test")
    dataset_math_train = load_math_reinforced()

    batch_size = 100
    logger = ExperimentLogger(
        results_file="../experiments/rveerara/"
        + str(num_shots)
        + "shot_reinforced.json",
        logging_frequency=batch_size,
    )
    question_ids_solved_already = logger.question_ids_solved
    print(f"Already solved {len(question_ids_solved_already)} questions")

    unsolved_questions = [
        instance
        for instance in dataset_math_test
        if instance["id"] not in question_ids_solved_already
    ]

    all_question_shots_from_math_train = [
        (instance["question"], instance["reasoning"], instance["answer"])
        for instance in dataset_math_train
    ]
    many_shot_prompt_str = create_and_return_manyshot_prompt_as_str(
        question_list=all_question_shots_from_math_train, num_shots=num_shots
    )
    final_input_prompts = [
        SOLVE_PROMPT.format(many_shot_prompt_str, d["question"])
        for d in unsolved_questions
    ]

    # print(final_input_prompts[0])

    # exit(0)

    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        download_dir="/home/rveerara/.cache/huggingface/hub",
        max_model_len=80000,
        gpu_memory_utilization=0.95,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=5120,
        stop_token_ids=[128001, 128008, 128009],
    )

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

            batch_results.append(
                {
                    "question_id": dataset_math_test[idx]["id"],
                    "predicted_answer": extracted["answer"],
                }
            )

            if not parsed:
                failed_count += 1
                continue

            try:
                gt_answer = float(dataset_math_test[idx]["answer"])
            except Exception:
                incorrect_count += 1
                continue

            if extracted["answer"] == gt_answer:
                correct_count += 1
            else:
                incorrect_count += 1

        total = correct_count + incorrect_count + failed_count
        print(f"Progress: {total}/{len(final_input_prompts)} questions processed")
        print(
            f"Correct: {correct_count}, Incorrect: {incorrect_count}, Unparsed: {failed_count}"
        )
        print(f"Accuracy: {correct_count/total:.2%}")
        logger.log_results(batch_results)
        logger.write_results_to_file()
