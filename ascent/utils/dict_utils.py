from collections.abc import MutableMapping
from typing import Any, Union


def flatten_dict(d: Union[dict[str, Any], MutableMapping]) -> dict[str, Any]:
    """Flatten a nested dictionary structure.

    Parameters:
        d: The input dictionary, which may contain nested dictionaries.

    Returns:
        A flattened dictionary.

    Example:
        Given the input dictionary:
        {
            'key1': {
                'subkey1': {'value': 1},
                'subkey2': {'value': 2}
            },
            'key2': {
                'subkey3': {'value': 3},
                'subkey4': {'value': 4}
            }
        }

        The output will be:
        {
            'subkey1': {'value': 1},
            'subkey2': {'value': 2},
            'subkey3': {'value': 3},
            'subkey4': {'value': 4}
        }
    """
    # Initialize an empty list to store key-value pairs
    items = []

    # Iterate through key-value pairs in the input dictionary
    for k, v in d.items():
        # If the value is a nested dictionary, recursively flatten it
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v).items())
        else:
            # If the value is not a dictionary, add the key-value pair to the list
            items.append((k, v))

    # Convert the list of key-value pairs to a dictionary and return it
    return dict(items)
