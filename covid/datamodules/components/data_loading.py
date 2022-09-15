import os
from pathlib import Path
from typing import Union


def get_case_identifiers_from_npz_folders(folder: Union[Path, str]) -> list[str]:
    case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith("npz")]
    return case_identifiers
