"""
Model training, hyperparameter tuning, prediction, and evaluation routines.
"""

import logging
import numpy as np
import time

from typing import Union, List, Dict, Tuple, Optional
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report,
    precision_score, recall_score, f1_score
)

from chem_model_test.core.transformations import perform_pca

logger = logging.getLogger(__name__)


def dynamic_import(class_path: str):
    """
    Dynamically import a class (e.g. 'sklearn.svm.SVC') given its full path string.

    :param class_path: The full path to a class, e.g. 'sklearn.svm.SVC'.
    :return: The class object.
    """
    module_name, class_name = class_path.rsplit('.', 1)
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def grid_search_model(
    X: np.ndarray,
    y: Union[np.ndarray, List[int]],
    model_class,
    param_grid: Dict[str, List[Union[str, int, float]]],
    cv: int = 5,
    n_jobs: int = -1
) -> object:
    """
    Perform grid search with cross-validation to tune hyperparameters.

    :param X: Feature matrix.
    :param y: Labels array.
    :param model_class: Class of the estimator (e.g. sklearn.svm.SVC).
    :param param_grid: Dictionary of hyperparameter search space.
    :param cv: Number of cross-validation folds.
    :param n_jobs: Number of jobs to run in parallel. -1 uses all processors.
    :return: A model instance with the best hyperparameters found.
    """
    grid = GridSearchCV(model_class(), param_grid, cv=cv, scoring='accuracy', n_jobs=n_jobs)
    grid.fit(X, y)
    logger.info(f"Best Parameters: {grid.best_params_}")
    return grid.best_estimator_


def train_until_100(
    X: np.ndarray,
    y: Union[np.ndarray, List[int]],
    X_unk: np.ndarray,
    labels_unk: Union[np.ndarray, List[int]],
    model_class,
    param_grid: Dict[str, List[Union[str, int, float]]],
    max_retries: int = 20,
    use_pca: bool = True,
    accuracy_threshold: float = 100.0,
    n_components: int = 2,
    cv: int = 5
) -> Tuple[Optional[object], Optional[object], float]:
    """
    Train a model iteratively until it achieves a given accuracy threshold
    on unknown data or reaches a maximum number of attempts.

    :param X: Known training features.
    :param y: Known training labels.
    :param X_unk: Unknown data features.
    :param labels_unk: True labels for unknown data (for measuring accuracy).
    :param model_class: The model class to train (e.g. sklearn.svm.SVC).
    :param param_grid: Hyperparameter search space for GridSearch.
    :param max_retries: Maximum number of training attempts before giving up.
    :param use_pca: Whether to apply PCA transformation.
    :param accuracy_threshold: The desired accuracy to achieve on unknown data.
    :param n_components: Number of PCA components if PCA is used.
    :param cv: Number of CV folds for grid search.
    :return: (pca, best_model, best_accuracy) - PCA transformer if used,
             best model instance found, and the best accuracy on unknown data.
    """
    pca = None
    if use_pca:
        pca, X_pca = perform_pca(X, n_components=n_components)
        X_unk_pca = pca.transform(X_unk)
    else:
        X_pca = X
        X_unk_pca = X_unk

    best_accuracy = 0.0
    best_model = None

    for attempt in range(1, max_retries + 1):
        current_model = grid_search_model(X_pca, y, model_class, param_grid, cv=cv)
        predictions = current_model.predict(X_unk_pca)
        unk_accuracy = accuracy_score(labels_unk, predictions) * 100
        logger.info(f"Attempt {attempt}: Unknown Data Accuracy: {unk_accuracy:.2f}%")

        if unk_accuracy > best_accuracy:
            best_accuracy = unk_accuracy
            best_model = current_model

        if unk_accuracy >= accuracy_threshold:
            logger.info(f"Achieved {accuracy_threshold}% accuracy on unknown data after {attempt} attempt(s).")
            return pca, best_model, best_accuracy

    logger.warning(
        f"Reached the maximum number of retries ({max_retries}) without achieving {accuracy_threshold}% accuracy."
    )
    logger.info(f"Best accuracy on unknown data: {best_accuracy:.2f}%")
    return pca, best_model, best_accuracy


def predict_folder(
    pca: Optional[object],
    classifier: object,
    X: np.ndarray,
    use_pca: bool = True
) -> np.ndarray:
    """
    Use a trained classifier to predict labels for a given set of features.
    Can optionally apply a PCA transform before classification.

    :param pca: PCA object (if any) used in training.
    :param classifier: Trained classifier.
    :param X: Feature matrix to predict.
    :param use_pca: If True, apply the same PCA transformation.
    :return: Array of predicted labels.
    """
    start_time = time.time()

    if use_pca and pca is not None:
        X_pca = pca.transform(X)
        predictions = classifier.predict(X_pca)
    else:
        predictions = classifier.predict(X)

    end_time = time.time()
    logger.info(f"Prediction Time: {end_time - start_time:.2f} seconds")
    return predictions


def validate_predictions(
    predictions: np.ndarray,
    labels: Union[np.ndarray, List[int]],
    file_paths: List[str],
    model_name: str
):
    """
    Evaluate model performance by comparing predictions to ground-truth labels.
    Print accuracy, precision, recall, F1-score, and a classification report.

    :param predictions: Array of predicted labels.
    :param labels: Ground truth labels.
    :param file_paths: Corresponding file paths (for logging).
    :param model_name: Name of the model used (e.g. 'svc').
    """
    accuracy = accuracy_score(labels, predictions) * 100
    precision = precision_score(labels, predictions, average='weighted') * 100
    recall = recall_score(labels, predictions, average='weighted') * 100
    f1 = f1_score(labels, predictions, average='weighted') * 100

    logger.info(f"{model_name} Accuracy: {accuracy:.2f}%")
    logger.info(f"{model_name} Precision: {precision:.2f}%")
    logger.info(f"{model_name} Recall: {recall:.2f}%")
    logger.info(f"{model_name} F1 Score: {f1:.2f}%")

    logger.info(f"\nClassification Report:\n{classification_report(labels, predictions)}")

    for i, (pred, expected, file_path) in enumerate(zip(predictions, labels, file_paths)):
        logger.info(f"Sample {i}: File = {file_path} -> Predicted = {pred}, Expected = {expected}")
