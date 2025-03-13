"""
Visualization and plotting utilities.
Functions for plotting PCA/t-SNE results, combined projections, etc.
"""

import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union, List, Optional, Callable
from datetime import datetime

from chem_model_test.core.transformations import perform_pca, perform_tsne

logger = logging.getLogger(__name__)

THIS_FILE = os.path.abspath(__file__)
CORE_DIR = os.path.dirname(THIS_FILE)
PACKAGE_ROOT = os.path.abspath(os.path.join(CORE_DIR, '..'))
PLOTS_DIR = os.path.join(PACKAGE_ROOT, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

def _make_timestamp() -> str:
    """
    Return a string timestamp, e.g. '20250313_192815'.
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def plot_predictions(
    known_data: np.ndarray,
    known_labels: Union[np.ndarray, List[int]],
    unknown_data: np.ndarray,
    unknown_predictions: Union[np.ndarray, List[int]],
    file_paths_unk: List[str],
    method: str = "pca",
    n_components: int = 2,
    perplexity: int = 30,
    accuracy: Optional[float] = None,
    extract_label_fn: Optional[Callable[[str, str], Optional[int]]] = None,
    pattern: str = r'(\d+)%',
    model_name: Optional[str] = None,  # <--- Model name from predict_flow
) -> None:
    """
    Plot a combined projection of known and unknown data,
    labeling unknown points with predicted + actual labels.
    Save a high-resolution SVG to chem_model_test/plots/ with a
    date/time stamp in the filename.

    :param known_data: Features of the known (training) data.
    :param known_labels: Labels of known data.
    :param unknown_data: Features of the unknown data.
    :param unknown_predictions: Predicted labels for the unknown data.
    :param file_paths_unk: Paths for unknown data (for extracting actual labels).
    :param method: Dimensionality reduction method ('pca' or 'tsne').
    :param n_components: Number of components for PCA/t-SNE.
    :param perplexity: Perplexity parameter for t-SNE.
    :param accuracy: Accuracy value to display in the plot title.
    :param extract_label_fn: Function to extract true labels from file paths.
    :param pattern: Regex pattern used by extract_label_fn.
    :param model_name: Name of the model used (e.g. 'svc', 'random_forest').
    """
    all_data = np.vstack([known_data, unknown_data])
    all_labels = np.hstack([known_labels, unknown_predictions])

    # Dimensionality reduction
    if method == "pca":
        _, projected = perform_pca(all_data, n_components)
    elif method == "tsne":
        projected = perform_tsne(all_data, n_components, perplexity)
    else:
        logger.error("Invalid method for plotting. Choose 'pca' or 'tsne'.")
        return

    known_projected = projected[:len(known_data)]
    unknown_projected = projected[len(known_data):]

    plt.figure(figsize=(10, 6))
    unique_known_labels = np.unique(known_labels)
    colors_known = plt.cm.jet(np.linspace(0, 1, len(unique_known_labels)))

    # Plot known data
    for i, label in enumerate(unique_known_labels):
        idx = (known_labels == label)
        plt.scatter(known_projected[idx, 0], known_projected[idx, 1],
                    color=colors_known[i], label=f'{label}%', alpha=0.8)

    # Plot unknown data with predicted vs. actual
    if extract_label_fn is None:
        def extract_label_fn(_fp: str, _pat: str) -> str:
            return "?"

    for i, (pred_label, file_path) in enumerate(zip(unknown_predictions, file_paths_unk)):
        correct_label = extract_label_fn(file_path, pattern)
        label_str = f"Pred: {pred_label}%, Actual: {correct_label}%"
        plt.scatter(unknown_projected[i, 0], unknown_projected[i, 1], color='gray', alpha=0.5)
        plt.text(unknown_projected[i, 0], unknown_projected[i, 1], label_str, fontsize=8)

    # Construct the plot title
    plot_title = f"Combined {method.upper()} Projection"
    if model_name is not None:
        plot_title += f" ({model_name})"
    if accuracy is not None:
        plot_title += f" - Accuracy: {accuracy:.2f}%"
    plt.title(plot_title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="Solute Concentration (%)")

    # Build the filename: combined_{model_name}_{method}_{timestamp}.svg
    timestamp = _make_timestamp()
    model_str = model_name or "NoModel"
    svg_filename = f"combined_{model_str}_{method}_{timestamp}.svg"

    svg_path = os.path.join(PLOTS_DIR, svg_filename)
    plt.savefig(svg_path, format="svg", dpi=300)
    logger.info(f"Saved high-res SVG to: {svg_path}")

    plt.show()
