"""
visualizations_v2.py

Plotting utilities for the new main script. 
Averaging data grouped by subdirectories, then plotting.
"""

import os
import logging
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

logger = logging.getLogger(__name__)

def plot_averaged_loss_profile_v2(
    subdir_dict: Dict[str, List[pd.DataFrame]],
    output_folder: str = "plots"
):
    """
    Given a dictionary of subdir -> list of DataFrames,
    compute row-wise averages for each subdir and plot them.

    :param subdir_dict: A dict { "10%": [df1, df2, ...], "20%": [df3, ...], ... }
    :param output_folder: Folder to save the resulting SVG plot.
    """
    if not subdir_dict:
        logger.info("No data provided for plotting.")
        return

    color_map = plt.get_cmap('tab20')
    plotted_folders = 0

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))

    # Sort subdirs by numeric value so we have a nice left-to-right legend order
    def subdir_int(sd_name: str):
        import re
        match = re.search(r"(\d+)", sd_name)
        return int(match.group(1)) if match else 0

    sorted_subdirs = sorted(subdir_dict.keys(), key=subdir_int)

    for idx, subdir in enumerate(sorted_subdirs):
        color = color_map(idx % 20)
        data_frames = subdir_dict[subdir]
        if not data_frames:
            logger.info(f"No valid DataFrames in subdir {subdir}.")
            continue

        # Concatenate them all
        big_df = pd.concat(data_frames, ignore_index=True)
        # Row-wise average: 
        #   if each DataFrame is the same shape with the same frequency axis
        #   you can do grouping by index. 
        #   If you need frequency alignment, you'll need a different merge approach.
        average_df = big_df.groupby(big_df.index).mean()

        # Expect columns: 0 => Frequency (Hz), 1 => Power (dB)
        if average_df.shape[1] < 2:
            logger.warning(f"Subdir {subdir} does not have >=2 columns.")
            continue

        freq_ghz = average_df.iloc[:, 0] / 1e9
        power_db = average_df.iloc[:, 1]

        plt.plot(freq_ghz, power_db, label=subdir, color=color)
        plotted_folders += 1

    if plotted_folders > 0:
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Power (dB)")
        plt.title(r'Averaged Loss Profile of $H_2O$ / $C_6H_{12}O_6$ Concentrations')
        plt.legend(title=r'$C_6H_{12}O_6$ Concentration')

        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, "averaged_loss_profile.svg")
        plt.savefig(output_path, format='svg')
        plt.show()
    else:
        logger.info("No folders plotted.")
