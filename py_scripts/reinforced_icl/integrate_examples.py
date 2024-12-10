import json

def load_reinforce(reinforce_filename):
    with open(reinforce_filename, "r") as f:
        data = f.read()
    json_data = json.loads(data)
    questions = []
    reasonings = []
    answers = []
    for set in json_data:
        for example in set:
            question = example["problem"]
            reasoning = example["reasoning"]
            answer = example["answer"]
            questions.append(question)
            reasonings.append(reasoning)
            answers.append(answer)
    return questions, reasonings, answers


def save_formatted(data_filename, questions, reasonings, answers):
    with open(data_filename, "w") as f:
        json.dump({"question": questions, "reasonings": reasonings, "answers": answers}, f)

if __name__ == "__main__":
    questions, reasonings, answers = load_reinforce("../../src/synth_data/cot_synth_data_70b_int4_3.json")
    save_formatted("../../datasets/MATH500/MATH_reinforced_numeric.json", questions, reasonings, answers)