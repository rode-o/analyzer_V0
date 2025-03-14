# chem_model_test/core/averaging.py

import logging
import numpy as np
from typing import Dict, List

logger = logging.getLogger(__name__)

def average_subdir_arrays(data_dict: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    """
    Given a dict { '10%': [arr1, arr2, ...], '20%': [arr3, ...], ... },
    where each arr is shape (N, 2) [frequency, power],
    compute the row-wise average for each subdirectory's arrays.

    Returns a dict { '10%': avg_array, '20%': avg_array, ... }
    where avg_array has shape (N, 2).
    """
    avg_dict = {}
    for subdir, arrays in data_dict.items():
        if not arrays:
            # No data in that subdir
            continue
        
        try:
            # stack shape: (#files, N, 2)
            stacked = np.stack(arrays, axis=0)
            # row-wise average => shape (N, 2)
            avg_array = stacked.mean(axis=0)
            avg_dict[subdir] = avg_array
        except ValueError as e:
            logger.error(f"Cannot stack arrays in subdir '{subdir}'. Mismatched shapes? {e}")
    
    return avg_dict
