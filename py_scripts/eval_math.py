import argparse
import json
import os
import pickle

import dspy

def load_math_as_examples(path="../datasets/MATH", split="train"):
    with open(os.path.join(path, split, "dataset.json")) as f:
        data = json.load(f)
    
    examples = []

    for question, answer in zip(data["question"], data["extracted_answers"]):
        example = dspy.Example(question=question, answer=answer).with_inputs("question")
        examples.append(example)

    return examples

class MathSignature(dspy.Signature):
    """Answer the math question."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="The final answer in latex format. Do not include the \\boxed{} symbol.")

class BaselineCoT(dspy.Module):

    def __init__(self):
        self.cot = dspy.ChainOfThought(MathSignature)
    
    def forward(self, example):
        response = self.cot(**example.inputs())
        return response

def run_eval():
    dataset = load_math_as_examples(split="test")
    print(f"Length of dataset: {len(dataset)}")

    baseline = dspy.ChainOfThought(MathSignature)

    # evaluator = dspy.evaluate.Evaluate(devset=dataset, num_threads=24, display_progress=True, display_table=True)
    evaluator = dspy.evaluate.Evaluate(devset=dataset, num_threads=24)

    score, each_scores, outputs = evaluator(baseline, metric=dspy.evaluate.metrics.answer_exact_match, return_all_scores=True, return_outputs=True)

    # with open("outputs.json", "w") as f:
    #     json.dump({
    #         "score": score,
    #         "scores": each_scores,
    #         "outputs": outputs
    #     }, f)

    # Save the outputs as a pickle file
    with open("outputs.pkl", "wb") as f:
        pickle.dump({
            "score": score,
            "scores": each_scores,
            "outputs": outputs
        }, f)

    print(f"Score: {score}")

# dspy.inspect_history()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, nargs='?', const='meta-llama/Llama-3.1-8B-Instruct', help="Large Language Model to use")
    parser.add_argument('--hostname', type=str, nargs='?', const='localhost', help="Hostname of the API")
    parser.add_argument('--port', type=int, nargs='?', const=8000, help="Port of the API")
    args = parser.parse_args()

    api_base = f"http://{args.hostname}:{args.port}/v1"
    api_key = "EMPTY"
    lm = dspy.LM("openai/" + "llm", api_base=api_base, api_key=api_key, cache=False)
    dspy.configure(lm=lm)

    run_eval()