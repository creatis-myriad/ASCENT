import os
from pathlib import Path
from typing import Union


def get_case_identifiers_from_npz_folders(folder: Union[Path, str]) -> list[str]:
    """Retrieve the case identifiers from (cropped or preprocessed) folder of .npz files. The
    filename should be named in nnUNet format, e.g. BraTS_0001.npz becomes BraTS_0001.

    Args:
        folder: Folder containing .npz files.

    Returns:
        List of strings containing all the case identifiers of the given folder.
    """
    case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith("npz")]
    return case_identifiers
