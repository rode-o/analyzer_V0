# chem_model_test/core/subdir_loader.py

import os
import logging
import re
from typing import Dict, List
import numpy as np
from .data_io import read_csv_as_np

logger = logging.getLogger(__name__)

def find_subdirectories(base_path: str) -> List[str]:
    """
    List subdirectories under `base_path`, sorted by the
    numeric percentage in their name (e.g. '10%', '20%', etc.).
    """
    if not os.path.isdir(base_path):
        logger.error(f"Directory not found: {base_path}")
        return []
    
    subdirs = [
        d for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
    ]

    def extract_int(s: str) -> int:
        match = re.search(r"(\d+)", s)
        return int(match.group(1)) if match else 0

    subdirs.sort(key=extract_int)
    return subdirs

def load_data_by_subdir(base_path: str) -> Dict[str, List[np.ndarray]]:
    """
    For each subdirectory (e.g. '10%', '20%', etc.), load all CSV files
    as NumPy arrays. Return a dict:
        { '10%': [array1, array2, ...], '20%': [...], ... }
    """
    data_dict = {}
    subdirs = find_subdirectories(base_path)

    for subdir in subdirs:
        folder_path = os.path.join(base_path, subdir)
        csv_files = [
            f for f in os.listdir(folder_path)
            if f.endswith(".csv")
        ]
        arr_list = []
        for csvf in csv_files:
            file_path = os.path.join(folder_path, csvf)
            arr = read_csv_as_np(file_path)
            if arr is not None:
                arr_list.append(arr)

        data_dict[subdir] = arr_list
    
    return data_dict
