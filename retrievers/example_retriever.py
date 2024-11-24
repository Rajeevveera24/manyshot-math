# External imports
import abc

class ExampleRetriever():

    def __init__(self, data_path):
        self.data_path = data_path
        self.load_data()
        self.organize_data()
        self.data_count = self.count_data()
    
    @abc.abstractmethod
    def load_data(self):
        ...
    
    @abc.abstractmethod
    def organize_data(self):
        ...

    @abc.abstractmethod
    def count_data(self):
        ...

    @abc.abstractmethod
    def retrieve(self, n_examples):
        ...