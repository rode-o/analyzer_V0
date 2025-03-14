"""
Data loading and collection utilities.

Contains functions for reading CSV files, walking the directory structure
to gather data, and extracting labels from file paths.
"""

import os
import re
import logging
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Union

logger = logging.getLogger(__name__)


def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load data from a CSV file given the file path.

    :param file_path: The path to the CSV file.
    :return: The loaded data as a pandas DataFrame, or None if an error occurs.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None

    try:
        data = pd.read_csv(file_path, header=0)
        logger.info(f"Data loaded from {file_path}, shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return None


def extract_label_from_path(path: str, pattern: str = r'(\d+)%') -> Optional[int]:
    """
    Extract a numeric label (e.g., 10, 20, etc.) from a file path
    by searching for a pattern like '10%', '30%', etc.

    :param path: The file path from which to extract the label.
    :param pattern: A regular expression pattern for extracting the label.
    :return: The integer label if found, otherwise None.
    """
    try:
        filename = os.path.basename(path)
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))
        else:
            logger.error(f"No valid numeric label found in {path}")
            return None
    except (ValueError, IndexError) as e:
        logger.error(f"Error extracting label from path {path}: {e}")
        return None


def collect_data(
    directory_path: str,
    file_extension: str = '.csv',
    extract_labels: bool = True,
    feature_index: int = 1,
    pattern: str = r'(\d+)%'
) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    """
    Recursively collect data from all .csv files under a given directory.
    Optionally extract labels from the file path and use a specific column as features.

    :param directory_path: Path to the directory containing CSV files.
    :param file_extension: Which file extension to look for (default '.csv').
    :param extract_labels: Whether to attempt to extract numeric labels from file paths.
    :param feature_index: Index of the feature column to be used from each CSV.
    :param pattern: Regex pattern used to extract label percentages from paths.
    :return: A tuple containing:
             - X: Feature matrix (n_samples x 1) if feature_index is used,
             - y: Optional array of labels if extract_labels is True and found,
             - file_paths: List of file paths processed.
    """
    X, y, file_paths = [], [], []
    if not os.path.exists(directory_path):
        logger.error(f"Directory not found: {directory_path}")
        return np.array([]), None, []

    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(file_extension):
                file_path = os.path.join(root, file)
                data = load_data(file_path)
                label = extract_label_from_path(file_path, pattern) if extract_labels else None

                if data is not None and not data.empty:
                    try:
                        # We use just one column as features; adapt as necessary.
                        X.append(data.iloc[:, feature_index].values)
                        file_paths.append(file_path)

                        if extract_labels and label is not None:
                            y.append(label)
                    except IndexError as e:
                        logger.error(f"Error processing file {file_path}: {e}")

    if X:
        # Stack all 1D arrays vertically and convert to float
        X = np.vstack(X).astype(float)
        if extract_labels and len(y) > 0:
            y = np.array(y)
        else:
            y = None
    else:
        # If no data was found, return empty
        X = np.array([])

    return X, y, file_paths


def collect_dataframes_by_subdirectory(
    directory_path: str,
    file_extension: str = '.csv',
    pattern: str = r'(\d+)%'
) -> dict:
    """
    Collect all CSV files under `directory_path`, grouped by the
    subdirectory name (e.g. "10%", "20%"), returning a dict:
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

        for file in csv_files:
            file_path = os.path.join(root, file)
            df = load_data(file_path)
            if df is not None and not df.empty:
                subdir_dict.setdefault(subdir_name, []).append(df)

    return subdir_dict
