import random


class nnUNet_Iterator:
    def __init__(self, list_of_dicts):
        self.list_of_dicts = list_of_dicts

    def __iter__(self):
        return self

    def __next__(self):
        random_dict = random.choice(self.list_of_dicts)  # nosec B311
        return random_dict
