import json
import pickle  # nosec B403


def load_pickle(file: str, mode: str = "rb"):  # nosec B301
    with open(file, mode) as f:
        a = pickle.load(f)
    return a


def write_pickle(obj, file: str, mode: str = "wb") -> None:  # nosec B301
    with open(file, mode) as f:
        pickle.dump(obj, f)


def load_json(file: str):
    with open(file) as f:
        a = json.load(f)
    return a


def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, "w") as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)
