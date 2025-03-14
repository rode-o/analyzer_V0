import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

def read_csv_as_np(file_path: str) -> np.ndarray:
    """
    Load a CSV file into a NumPy array.
    Assumes there's exactly one header row (skiprows=1),
    and we only keep columns 0 and 1:
      column 0 => frequency (Hz)
      column 1 => power (dB)

    Returns:
        A 2D NumPy array of shape (N, 2), or None on error.
    """
    if not os.path.isfile(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    
    try:
        # load only the first 2 columns
        data = np.loadtxt(file_path, delimiter=",", skiprows=1, usecols=(0,1))
        # data now has shape (N, 2)
        return data
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None
