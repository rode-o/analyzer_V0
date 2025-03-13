import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def list_subdirectories(base_path='trData'):
    """
    Lists all subdirectories within a specified base directory, sorted numerically based on the percentage value in the name.
    
    Args:
        base_path (str): The path to the directory from which subdirectories are listed.

    Returns:
        list: A list of sorted subdirectory names or an empty list if an error occurs.
    """
    try:
        subdirectories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        subdirectories.sort(key=lambda x: int(x.replace('%', '')))
        logging.info(f"Sorted subdirectories: {subdirectories}")
        return subdirectories
    except FileNotFoundError:
        logging.error(f"The directory {base_path} does not exist.")
        return []
    except PermissionError:
        logging.error(f"Permission denied to access {base_path}.")
        return []

def find_csv_files(subdirectories, base_path='trData'):
    """
    Finds all CSV files within each subdirectory under the base path using os.walk for better efficiency.
    
    Args:
        subdirectories (list): List of subdirectory names.
        base_path (str): The base path where the subdirectories are located.

    Returns:
        list: A list of tuples containing the subdirectory and file path.
    """
    files_info = []
    for subdir in subdirectories:
        path = os.path.join(base_path, subdir)
        for root, dirs, files in os.walk(path):
            csv_files = [os.path.join(root, file) for file in files if file.endswith('.csv')]
            files_info.extend([(subdir, file) for file in csv_files])
    return files_info

def plot_average_data(files_info):
    """
    Plots the row-wise average data from multiple CSV files in each subdirectory, using the first two columns, 
    and scales the x-axis values from Hz to GHz by dividing by 1e9.
    
    Args:
        files_info (dict): Dictionary with subdirectories as keys and list of file paths as values.
    """
    color_map = plt.get_cmap('tab20')
    subdir_colors = {}
    plotted_folders = 0

    plt.figure(figsize=(10, 6))
    ax = plt.gca()  # Get the current axis
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))  # Set formatter to avoid scientific notation

    for idx, (subdir, files) in enumerate(files_info.items()):
        subdir_colors[subdir] = color_map(idx % 20)

        # Read and collect data only if there are at least 2 columns in the CSV
        data_frames = []
        for file in files:
            df = pd.read_csv(file)
            if df.shape[1] >= 2:
                data_frames.append(df)

        if data_frames:
            aligned_df = pd.concat(data_frames, axis=0)
            average_df = aligned_df.groupby(aligned_df.index).mean()
            plt.plot(average_df.iloc[:, 0] / 1e9, average_df.iloc[:, 1], 
                     label=subdir, color=subdir_colors[subdir])
            plotted_folders += 1
        else:
            logging.info(f"No valid data found in {subdir} to plot.")

    if plotted_folders > 0:
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Power (dB)")
        plt.title(r'Averaged Loss Profile of $H_2O$ / $C_6H_{12}O_6$ Concentrations')
        plt.legend(title=r'$C_6H_{12}O_6$ Concentration')

        # Create top-level folder (e.g. "plots") to save the SVG
        output_folder = "plots"
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, "averaged_loss_profile.svg")
        
        plt.savefig(output_path, format='svg')
        plt.show()
    else:
        logging.info("No data plotted.")

def main():
    subdirectories = list_subdirectories()
    files_info = find_csv_files(subdirectories)

    # Group the CSV files by their subdirectory
    files_dict = {}
    for subdir, file in files_info:
        if subdir not in files_dict:
            files_dict[subdir] = []
        files_dict[subdir].append(file)

    plot_average_data(files_dict)

if __name__ == '__main__':
    main()
