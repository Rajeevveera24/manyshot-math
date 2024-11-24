# External imports
import numpy as np

# Internal imports
from random_retriever import RandomRetriever

class RandomRetrieverMath(RandomRetriever):

    def __init__(self, data_path):
        super().__init__(data_path)

    def count_data(self):
        self.instances = len(self.questions)

    def organize_data(self):
        self.questions = self.dataset["question"]
        self.answers = self.dataset["extracted_answers"]

    def retrieve(self, n_examples):
        selected_indices = np.random.choice(self.instances, (n_examples), replace=False)
        selected_questions = np.array(self.questions)[selected_indices].tolist()
        selected_answers = np.array(self.answers)[selected_indices].tolist()
        return selected_questions, selected_answers