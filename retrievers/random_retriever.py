# External imports
import json

# Internal imports
from example_retriever import ExampleRetriever

class RandomRetriever(ExampleRetriever):

    def __init__(self, data_path):
        super().__init__(data_path)
    
    def load_data(self):
        with open(self.data_path, "r") as data_file:
            raw_dataset = data_file.read()
        self.dataset = json.loads(raw_dataset)