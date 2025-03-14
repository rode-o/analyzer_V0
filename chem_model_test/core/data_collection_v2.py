"""
data_collection_v2.py

Provides data collection methods for the new main script,
grouping CSV data by subdirectory for row-wise averaging/plotting.
"""

import os
import re
import logging
import pandas as pd
from typing import Dict, List
from chem_model_test.core.data_loading import load_data  # Reuse existing function

logger = logging.getLogger(__name__)

def collect_dataframes_by_subdirectory_v2(
    directory_path: str,
    file_extension: str = '.csv',
    pattern: str = r'(\d+)%'
) -> Dict[str, List[pd.DataFrame]]:
    """
    Collect all CSV files under `directory_path`, grouped by the
    subdirectory name (e.g., "10%", "20%"), returning a dict:
        {"10%": [df1, df2, ...], "20%": [df1, df2, ...], ...}

    :param directory_path: Path to the directory containing CSV files.
    :param file_extension: Which file extension to look for (default '.csv').
    :param pattern: Regex pattern used to validate or parse subdir names (optional).
    :return: A dict mapping subdir (e.g., "10%") -> list of DataFrames.
    """
    subdir_dict = {}
    if not os.path.exists(directory_path):
        logger.error(f"Directory not found: {directory_path}")
        return subdir_dict

    for root, _, files in os.walk(directory_path):
        # The subdirectory name could be "10%", "20%", etc.
        subdir_name = os.path.basename(root)
        csv_files = [f for f in files if f.endswith(file_extension)]

        # Skip if no CSV files in this folder
        if not csv_files:
            continue

        # Optionally check if the subdir matches the expected pattern:
        if not re.search(pattern, subdir_name):
            # Not a recognized subdirectory label, or top-level folder
            # You might skip or handle differently
            continue

        # For each CSV file in this subdir, load it and store it
        for file in csv_files:
            file_path = os.path.join(root, file)
            df = load_data(file_path)  # imported from your existing data_loading.py
            if df is not None and not df.empty:
                subdir_dict.setdefault(subdir_name, []).append(df)

    return subdir_dict
