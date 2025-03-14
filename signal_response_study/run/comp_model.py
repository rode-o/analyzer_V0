"""
Script for performing an end-to-end training and prediction flow, including
hyperparameter tuning and evaluation on unknown data. No hard-coded absolute
paths: we compute the 'trData' and 'unkData' directories relative to this script.
"""

import os
import logging
from sklearn.metrics import accuracy_score

from signal_response_study.comp_model_core.utils import setup_logging
from signal_response_study.comp_model_core.config import model_hyperparameters
from signal_response_study.comp_model_core.data_loading import collect_data, extract_label_from_path
from signal_response_study.comp_model_core.modeling import (
    dynamic_import, train_until_100, grid_search_model,
    predict_folder, validate_predictions
)
from signal_response_study.comp_model_core.transformations import perform_pca
from signal_response_study.comp_model_core.visualizations import plot_predictions


def predict_flow(training_dir: str, unknown_dir: str):
    """
    Perform model training and prediction based on user input. Collects
    training data, collects unknown data, asks for model choices, and
    trains until a certain accuracy threshold or for a fixed number of iterations.

    :param training_dir: Directory containing labeled training data.
    :param unknown_dir: Directory containing unlabeled test data (with hidden labels).
    """
    use_pca_str = input("Use PCA for prediction? (yes/no, default: no): ").strip().lower() or 'no'
    use_pca = (use_pca_str == 'yes')

    n_components = 2
    if use_pca:
        try:
            nc_str = input("Specify number of PCA components (default: 2): ").strip()
            n_components = int(nc_str) if nc_str else 2
        except ValueError:
            n_components = 2

    # Model selection
    model_choice = input("Choose a model (svc, decision_tree, random_forest, knn): ").strip().lower()
    if model_choice not in model_hyperparameters:
        print("Invalid model choice. Please choose a valid model from config.")
        return

    model_info = model_hyperparameters[model_choice]
    class_path = model_info['model_class']
    ModelClass = dynamic_import(class_path)
    param_grid = model_info['param_grid']

    # Load training data
    X, y, file_paths_tr = collect_data(training_dir)
    if X.size == 0:
        logging.warning("No training data found or folder is empty: %s", training_dir)
        return

    # Load unknown data
    X_unk, labels_unk, file_paths_unk = collect_data(
        unknown_dir, file_extension='.csv', extract_labels=True
    )
    if X_unk.size == 0:
        logging.warning("No unknown data found or folder is empty: %s", unknown_dir)
        return

    # Choose training mode
    training_mode = input(
        "Train until 100% accuracy (type '100%') or specify 'iter'? (default: '100%'): "
    ).strip().lower() or '100%'
    clf = None
    best_accuracy = 0.0
    pca = None

    if training_mode == '100%':
        max_retries = int(input("Max attempts to achieve 100% accuracy (default: 20): ").strip() or 20)
        accuracy_threshold = float(input("Desired accuracy threshold (default: 100.0): ").strip() or 100.0)

        pca, clf, best_accuracy = train_until_100(
            X, y, X_unk, labels_unk,
            ModelClass, param_grid,
            max_retries=max_retries,
            use_pca=use_pca,
            accuracy_threshold=accuracy_threshold,
            n_components=n_components
        )

    elif training_mode == 'iter':
        num_iterations = int(input("Number of iterations (default: 10): ").strip() or 10)

        # If using PCA, transform once before multiple training attempts
        if use_pca:
            pca, X_pca = perform_pca(X, n_components=n_components)
            X_unk_pca = pca.transform(X_unk)
        else:
            X_pca = X
            X_unk_pca = X_unk

        best_model = None
        for iteration in range(1, num_iterations + 1):
            current_model = grid_search_model(X_pca, y, ModelClass, param_grid, cv=5)
            predictions = current_model.predict(X_unk_pca)
            unk_accuracy = accuracy_score(labels_unk, predictions) * 100
            logging.info(f"Iteration {iteration}: Accuracy on unknown data: {unk_accuracy:.2f}%")

            if unk_accuracy > best_accuracy:
                best_accuracy = unk_accuracy
                best_model = current_model

        clf = best_model

    else:
        logging.warning("Invalid input. Please specify '100%' or 'iter'.")
        return

    # Predict on unknown data with the final (best) model
    predictions = predict_folder(pca, clf, X_unk, use_pca)
    validate_predictions(predictions, labels_unk, file_paths_unk, model_choice)

    # Optionally plot combined projection
    combined_plot_choice = input(
        "Plot combined PCA/t-SNE projection after predictions? (yes/no, default: yes): "
    ).strip().lower() or 'yes'
    if combined_plot_choice == 'yes':
        plot_method = input("Choose 'pca' or 'tsne' (default: 'pca'): ").strip().lower() or 'pca'
        perplexity = 30
        if plot_method == 'tsne':
            try:
                perplexity = int(input("Specify t-SNE perplexity (default: 30): ").strip() or 30)
            except ValueError:
                perplexity = 30

        # Pass model_choice so the final plot file name and title reflect the model
        plot_predictions(
            known_data=X,
            known_labels=y,
            unknown_data=X_unk,
            unknown_predictions=predictions,
            file_paths_unk=file_paths_unk,
            method=plot_method,
            n_components=n_components,
            perplexity=perplexity,
            accuracy=best_accuracy,
            extract_label_fn=extract_label_from_path,
            model_name=model_choice
        )


def main():
    """
    Main function for the predict_flow script. Sets up logging, then computes
    the relative paths to 'trData' and 'unkData' automatically based on this script's location.
    """
    setup_logging()

    # Path of the current file (predict_flow.py)
    this_file = os.path.abspath(__file__)  
    # .../Desktop/analyzer_v0/chem_model_test/entries/predict_flow.py

    # "entries/" folder
    entries_dir = os.path.dirname(this_file)  
    # .../Desktop/analyzer_v0/chem_model_test/entries

    # The parent folder is "chem_model_test/"
    package_root = os.path.abspath(os.path.join(entries_dir, '..'))
    # .../Desktop/analyzer_v0/chem_model_test

    # Build paths to 'trData' and 'unkData' from package_root
    training_dir = os.path.join(package_root, 'trData')
    unknown_dir = os.path.join(package_root, 'unkData')

    logging.info(f"Calculated training data path: {training_dir}")
    logging.info(f"Calculated unknown data path:  {unknown_dir}")

    predict_flow(training_dir, unknown_dir)


if __name__ == "__main__":
    main()
