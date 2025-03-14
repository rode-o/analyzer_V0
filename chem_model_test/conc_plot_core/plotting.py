# chem_model_test/core/plotting.py

import os
import logging
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict
from datetime import datetime

logger = logging.getLogger(__name__)

def plot_averaged_data(
    avg_dict: Dict[str, np.ndarray],
    output_svg: str = "averaged_loss_profile.svg"
):
    """
    Plot the averaged data from each subdir. Expects something like:
      {
        '10%': (N, 2) array => freq_Hz, power_dB,
        '20%': (N, 2) array => ...
        ...
      }

    If 'output_svg' is:
      - A directory (e.g. "plots"), the function saves to:
          "plots/averaged_loss_profile_YYYYMMDD_HHMMSS.svg"
      - A filename (e.g. "plots/custom_name.svg"), it saves exactly to that file name.

    The figure is then shown (plt.show()) and the path is logged.
    """
    if not avg_dict:
        logger.info("No averaged data to plot.")
        return
    
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))

    # Sort subdirs by numeric value so the legend is in ascending order
    def extract_int(s: str) -> int:
        match = re.search(r"(\d+)", s)
        return int(match.group(1)) if match else 0
    
    color_map = plt.get_cmap('tab20')
    sorted_subdirs = sorted(avg_dict.keys(), key=extract_int)

    for idx, subdir in enumerate(sorted_subdirs):
        arr = avg_dict[subdir]
        # arr[:, 0] => freq (Hz), arr[:, 1] => power (dB)
        freq_ghz = arr[:, 0] / 1e9
        power_db = arr[:, 1]
        color = color_map(idx % 20)

        plt.plot(freq_ghz, power_db, label=subdir, color=color)

    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Power (dB)")
    plt.title(r'Averaged Loss Profile of $H_2O$ / $C_6H_{12}O_6$ Concentrations')
    plt.legend(title=r'$C_6H_{12}O_6$ Concentration')

    # Check if output_svg is actually a directory or a filename
    # 1) If it's a directory (existing or not), we build a timestamped name inside it.
    # 2) If it's a file path ending in .svg, we use it directly.
    if output_svg.lower().endswith('.svg'):
        # It's a file path, use it as-is
        output_path = output_svg
    else:
        # It's presumably a directory, so we add a timestamped file name
        if not os.path.exists(output_svg):
            os.makedirs(output_svg, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"averaged_loss_profile_{timestamp}.svg"
        output_path = os.path.join(output_svg, file_name)

    # Create parent dirs for output_path if needed
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Save as an SVG, with high 'dpi'
    plt.savefig(output_path, format="svg", dpi=300)
    plt.show()

    logger.info(f"Saved plot to {output_path}")
