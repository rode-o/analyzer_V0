"""
Script for plotting data using PCA or t-SNE. Demonstrates usage of the core libraries
to load data, perform transformations, and visualize the results.
"""

import os
import logging
from chem_model_test.core.utils import setup_logging
from chem_model_test.core.data_loading import collect_data
from chem_model_test.core.transformations import perform_pca, perform_tsne
from chem_model_test.core.visualizations import plot_data

def plot_flow(training_dir: str):
    """
    Load data from the training_dir, then prompt the user to choose
    a dimensionality reduction technique and plot the results.
    
    :param training_dir: Path to directory containing training data.
    """
    X, y, _ = collect_data(training_dir)
    if X.size == 0:
        logging.warning("No data available for plotting.")
        return

    method_choice = input("Choose transformation ('pca' or 'tsne', default 'pca'): ").strip().lower() or 'pca'
    if method_choice == 'pca':
        _, X_proj = perform_pca(X, n_components=2)
        plot_data(X_proj, y, title="PCA Projection", legend_title="Label (%)")
    elif method_choice == 'tsne':
        X_proj = perform_tsne(X, n_components=2, perplexity=30)
        plot_data(X_proj, y, title="t-SNE Projection", legend_title="Label (%)")
    else:
        logging.warning("Invalid choice. No plot generated.")


def main():
    """
    Main function for the plot_flow script. Sets up logging and runs the plot routine.
    """
    setup_logging()
    base_dir = os.getcwd()
    training_dir = os.path.join(base_dir, 'trData')
    plot_flow(training_dir)


if __name__ == "__main__":
    main()
