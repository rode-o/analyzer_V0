# chem_model_test/entries/plot_flow_clean.py

"""
plot_flow_clean.py

Demonstrates a modular NumPy-based approach for row-wise averaging
and plotting frequency/power data from CSV files.
"""

import logging
import os

from chem_model_test.conc_plot_core.subdir_loader import load_data_by_subdir
from chem_model_test.conc_plot_core.averaging import average_subdir_arrays
from chem_model_test.conc_plot_core.plotting import plot_averaged_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Build path to "trData" based on the location of this file, if desired:
    script_dir = os.path.dirname(__file__)
    tr_data_path = os.path.join(script_dir, "..", "trData")
    tr_data_path = os.path.abspath(tr_data_path)

    # 1) Load CSV arrays from each subdirectory
    data_dict = load_data_by_subdir(tr_data_path)

    # 2) Row-wise average arrays within each subdirectory
    avg_dict = average_subdir_arrays(data_dict)

    # 3) Plot the data
    #   Save the figure in chem_model_test/plots, for example.
    plot_path = os.path.join(script_dir, "..", "plots", "averaged_loss_profile.svg")
    plot_path = os.path.abspath(plot_path)
    plot_averaged_data(avg_dict, output_svg=plot_path)

if __name__ == "__main__":
    main()
