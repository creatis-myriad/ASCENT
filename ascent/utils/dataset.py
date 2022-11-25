import random


class nnUNet_Iterator:
    """Infinite iterable dataset for nnUNet's pipeline,"""

    def __init__(self, list_of_dicts: list[dict[str, str]]):
        """Initialize class instance.

        Args:
            list of dict: List of dictionaries containing the path to the preprocessed .npy data.
        """
        self.list_of_dicts = list_of_dicts

    def __iter__(self):  # noqa: D102
        return self

    def __next__(self):  # noqa: D102
        # draw a random dictionary from the list
        random_dict = random.choice(self.list_of_dicts)  # nosec B311
        return random_dict
