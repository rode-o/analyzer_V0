import os
import re
import time
import numpy as np
import pandas as pd
import logging
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define model parameters for grid search
model_hyperparameters = {
    'svc': {
        'model_class': SVC,
        'param_grid': {
            'kernel': ['linear', 'rbf'],
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        }
    },
    'decision_tree': {
        'model_class': DecisionTreeClassifier,
        'param_grid': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 20]
        }
    },
    'random_forest': {
        'model_class': RandomForestClassifier,
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10, 20],
            'criterion': ['gini', 'entropy']
        }
    },
    'knn': {
        'model_class': KNeighborsClassifier,
        'param_grid': {
            'n_neighbors': [3, 5, 7, 10],
            'weights': ['uniform', 'distance']
        }
    }
}


def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load data from a CSV file given the file path.

    Args:
        file_path (str): The path to the CSV file to be loaded.

    Returns:
        Optional[pd.DataFrame]: The loaded data in a pandas DataFrame, or None if loading fails.
    """
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return None
    try:
        data = pd.read_csv(file_path, header=0)
        logging.info(f"Data loaded from {file_path}, shape: {data.shape}")
        return data
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        return None


def perform_pca(X: np.ndarray, n_components: int = 2) -> Tuple[PCA, np.ndarray]:
    """
    Perform Principal Component Analysis (PCA) on the input data.

    Args:
        X (np.ndarray): Feature matrix to perform PCA on.
        n_components (int, optional): Number of principal components. Defaults to 2.

    Returns:
        Tuple[PCA, np.ndarray]: The fitted PCA object and the transformed data.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    logging.info(f"PCA performed with {n_components} components")
    return pca, X_pca


def perform_tsne(X: np.ndarray, n_components: int = 2, perplexity: int = 30) -> np.ndarray:
    """
    Perform t-SNE on the input data.

    Args:
        X (np.ndarray): Feature matrix to perform t-SNE on.
        n_components (int, optional): Number of components. Defaults to 2.
        perplexity (int, optional): t-SNE perplexity parameter. Defaults to 30.

    Returns:
        np.ndarray: Transformed data using t-SNE.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    logging.info(f"t-SNE performed with {n_components} components and perplexity {perplexity}")
    return X_tsne


def plot_data(projected: np.ndarray, labels: Union[np.ndarray, List[int]], figsize: Tuple[int, int] = (10, 6), title: str = "Projection", legend_title: str = "Label") -> None:
    """
    Plot the projected data (PCA or t-SNE) in 2D.

    Args:
        projected (np.ndarray): Transformed data matrix for plotting.
        labels (Union[np.ndarray, List[int]]): Corresponding labels of the data points.
        figsize (Tuple[int, int], optional): Size of the plot figure. Defaults to (10, 6).
        title (str, optional): Title of the plot. Defaults to "Projection".
        legend_title (str, optional): Title for the legend. Defaults to "Label".
    """
    plt.figure(figsize=figsize)
    unique_labels = np.unique(labels)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        idx = labels == label
        plt.scatter(projected[idx, 0], projected[idx, 1], color=colors[i], label=f'{label}%')

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title=legend_title)
    plt.show()


def plot_predictions(
    known_data: np.ndarray,
    known_labels: Union[np.ndarray, List[int]],
    unknown_data: np.ndarray,
    unknown_predictions: Union[np.ndarray, List[int]],
    file_paths_unk: List[str],
    method: str = "pca",
    n_components: int = 2,
    perplexity: int = 30,
    accuracy: Optional[float] = None
) -> None:
    """
    Plot training data alongside predicted unknown data using PCA or t-SNE.

    Args:
        known_data (np.ndarray): Feature matrix of known training data.
        known_labels (Union[np.ndarray, List[int]]): Labels of the known training data.
        unknown_data (np.ndarray): Feature matrix of the unknown data.
        unknown_predictions (Union[np.ndarray, List[int]]): Predicted labels of the unknown data.
        file_paths_unk (List[str]): Paths to the files containing the unknown data.
        method (str, optional): Dimensionality reduction method to use ('pca' or 'tsne'). Defaults to "pca".
        n_components (int, optional): Number of components for PCA/t-SNE. Defaults to 2.
        perplexity (int, optional): Perplexity parameter for t-SNE. Defaults to 30.
        accuracy (Optional[float], optional): Accuracy rating to display on the plot. Defaults to None.
    """
    pattern = r'(\d+)%'

    # Combine all data for dimensionality reduction
    all_data = np.vstack([known_data, unknown_data])
    all_labels = np.hstack([known_labels, unknown_predictions])

    if method == "pca":
        _, projected = perform_pca(all_data, n_components)
    elif method == "tsne":
        projected = perform_tsne(all_data, n_components, perplexity)
    else:
        logging.error("Invalid method for plotting. Choose 'pca' or 'tsne'.")
        return

    # Split the reduced data into known and unknown subsets
    known_projected = projected[:len(known_data)]
    unknown_projected = projected[len(known_data):]

    # Plot the known data
    plt.figure(figsize=(10, 6))
    unique_known_labels = np.unique(known_labels)
    colors_known = plt.cm.jet(np.linspace(0, 1, len(unique_known_labels)))
    for i, label in enumerate(unique_known_labels):
        idx = known_labels == label
        plt.scatter(known_projected[idx, 0], known_projected[idx, 1], color=colors_known[i], label=f'{label}%', alpha=0.8)

    # Plot unknown data with predicted and actual labels
    for i, (predicted_label, file_path) in enumerate(zip(unknown_predictions, file_paths_unk)):
        correct_label = extract_label_from_path(file_path, pattern)
        label_str = f"Pred: {predicted_label}%, Actual: {correct_label}%"
        plt.scatter(unknown_projected[i, 0], unknown_projected[i, 1], color='gray', alpha=0.5)
        plt.text(unknown_projected[i, 0], unknown_projected[i, 1], label_str, fontsize=8)

    title = f"Combined {method.upper()} Projection"
    if accuracy is not None:
        title += f" (Prediction Accuracy: {accuracy:.2f}%)"
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="Solute Concentration (%)")
    plt.show()


def grid_search_model(X: np.ndarray, y: Union[np.ndarray, List[int]], model_class: type, param_grid: Dict[str, List[Union[str, int, float]]], cv: int = 5, n_jobs: int = -1) -> object:
    """
    Optimize model parameters using grid search with cross-validation.

    Args:
        X (np.ndarray): Feature matrix for training.
        y (Union[np.ndarray, List[int]]): Labels corresponding to the training data.
        model_class (type): The model class to be trained.
        param_grid (Dict[str, List[Union[str, int, float]]]): Hyperparameter grid to search.
        cv (int, optional): Number of folds for cross-validation. Defaults to 5.
        n_jobs (int, optional): Number of parallel jobs for GridSearchCV. Defaults to -1.

    Returns:
        object: The model instance with the best hyperparameters.
    """
    grid = GridSearchCV(model_class(), param_grid, cv=cv, scoring='accuracy', n_jobs=n_jobs)
    grid.fit(X, y)
    logging.info(f"Best Parameters: {grid.best_params_}")
    return grid.best_estimator_


def train_until_100(X: np.ndarray, y: Union[np.ndarray, List[int]], X_unk: np.ndarray, labels_unk: Union[np.ndarray, List[int]], model_class: type, param_grid: Dict[str, List[Union[str, int, float]]], max_retries: int = 20, use_pca: bool = True, accuracy_threshold: float = 100.0, n_components: int = 2, cv: int = 5) -> Tuple[Optional[PCA], Optional[object], float]:
    """
    Train the model iteratively until achieving a target accuracy on unknown data or the retry limit is reached.

    Args:
        X (np.ndarray): Feature matrix of training data.
        y (Union[np.ndarray, List[int]]): Labels corresponding to the training data.
        X_unk (np.ndarray): Feature matrix of unknown data.
        labels_unk (Union[np.ndarray, List[int]]): Expected labels of the unknown data.
        model_class (type): The model class to be trained.
        param_grid (Dict[str, List[Union[str, int, float]]]): Hyperparameter grid to search.
        max_retries (int, optional): Maximum number of attempts. Defaults to 20.
        use_pca (bool, optional): Whether to use PCA for dimensionality reduction. Defaults to True.
        accuracy_threshold (float, optional): Accuracy target to achieve on unknown data. Defaults to 100.0.
        n_components (int, optional): Number of PCA components if PCA is used. Defaults to 2.
        cv (int, optional): Number of folds for cross-validation. Defaults to 5.

    Returns:
        Tuple[Optional[PCA], Optional[object], float]: The fitted PCA instance, the best model, and the best accuracy achieved.
    """
    if use_pca:
        pca, X_pca = perform_pca(X, n_components=n_components)
        X_unk_pca = pca.transform(X_unk)
    else:
        pca = None
        X_pca = X
        X_unk_pca = X_unk

    best_accuracy = 0.0
    best_model = None

    for attempt in range(1, max_retries + 1):
        current_model = grid_search_model(X_pca, y, model_class, param_grid, cv=cv)

        predictions = current_model.predict(X_unk_pca)
        unk_accuracy = accuracy_score(labels_unk, predictions) * 100
        logging.info(f"Attempt {attempt}: Unknown Data Accuracy: {unk_accuracy:.2f}%")

        if unk_accuracy > best_accuracy:
            best_accuracy = unk_accuracy
            best_model = current_model

        if unk_accuracy >= accuracy_threshold:
            logging.info(f"Achieved {accuracy_threshold}% accuracy on unknown data after {attempt} attempt(s).")
            return pca, best_model, best_accuracy

    logging.warning(f"Reached the maximum number of retries ({max_retries}) without achieving {accuracy_threshold}% accuracy.")
    logging.info(f"Best accuracy achieved on unknown data: {best_accuracy:.2f}%")
    return pca, best_model, best_accuracy


def predict_folder(pca: Optional[PCA], classifier: object, X: np.ndarray, use_pca: bool = True) -> np.ndarray:
    """
    Predict the classes of samples using a classifier, optionally with PCA transformation.

    Args:
        pca (Optional[PCA]): The fitted PCA instance for transformation. None if not applicable.
        classifier (object): The trained classifier for making predictions.
        X (np.ndarray): Feature matrix to predict.
        use_pca (bool, optional): Whether to use PCA transformation. Defaults to True.

    Returns:
        np.ndarray: Array of predicted labels.
    """
    start_time = time.time()

    if use_pca and pca is not None:
        X_pca = pca.transform(X)
        predictions = classifier.predict(X_pca)
    else:
        predictions = classifier.predict(X)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Prediction Time: {elapsed_time:.2f} seconds")
    return predictions


def extract_label_from_path(path: str, pattern: str = r'(\d+)%') -> Optional[int]:
    """
    Extract the numeric class percentage from a folder or file structure path using regular expressions.

    Args:
        path (str): The path from which to extract the numeric label.
        pattern (str, optional): Regular expression pattern to identify numeric labels. Defaults to r'(\d+)%'.

    Returns:
        Optional[int]: Extracted numeric label, or None if not found or invalid.
    """
    try:
        filename = os.path.basename(path)
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))
        else:
            logging.error(f"No valid numeric label found in {path}")
            return None
    except (ValueError, IndexError) as e:
        logging.error(f"Error extracting label from path {path}: {e}")
        return None


def collect_data(directory_path: str, file_extension: str = '.csv', extract_labels: bool = True, feature_index: int = 1, pattern: str = r'(\d+)%') -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    """
    Collect data from all files with a given extension within a directory structure.

    Args:
        directory_path (str): Path to the directory to search for files.
        file_extension (str, optional): File extension to search for. Defaults to '.csv'.
        extract_labels (bool, optional): Whether to extract labels based on file/folder names. Defaults to True.
        feature_index (int, optional): Index of the feature column to use. Defaults to 1.
        pattern (str, optional): Regular expression pattern for extracting numeric labels. Defaults to r'(\d+)%'.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray], List[str]]: The feature matrix, labels (if applicable), and a list of file paths.
    """
    X, y, file_paths = [], [], []
    if not os.path.exists(directory_path):
        logging.error(f"Directory not found: {directory_path}")
        return np.array([]), None, []

    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(file_extension):
                file_path = os.path.join(root, file)
                data = load_data(file_path)
                label = extract_label_from_path(file_path, pattern) if extract_labels else None
                if data is not None and not data.empty:
                    try:
                        X.append(data.iloc[:, feature_index].values)
                        file_paths.append(file_path)

                        if extract_labels and label is not None:
                            y.append(label)
                    except IndexError as e:
                        logging.error(f"Error processing file {file_path}: {e}")

    if X:
        X = np.vstack(X).astype(float)
        y = np.array(y) if extract_labels else None

    return X, y, file_paths


def validate_predictions(predictions: np.ndarray, labels: Union[np.ndarray, List[int]], file_paths: List[str], model_name: str):
    """
    Compare predictions with expected labels and display accuracy.

    Args:
        predictions (np.ndarray): Predicted labels.
        labels (Union[np.ndarray, List[int]]): Expected labels.
        file_paths (List[str]): Paths to the files corresponding to each sample.
        model_name (str): Name of the model used for prediction.
    """
    accuracy = accuracy_score(labels, predictions) * 100
    precision = precision_score(labels, predictions, average='weighted') * 100
    recall = recall_score(labels, predictions, average='weighted') * 100
    f1 = f1_score(labels, predictions, average='weighted') * 100

    logging.info(f"{model_name} Accuracy: {accuracy:.2f}%")
    logging.info(f"{model_name} Precision: {precision:.2f}%")
    logging.info(f"{model_name} Recall: {recall:.2f}%")
    logging.info(f"{model_name} F1 Score: {f1:.2f}%")

    logging.info(f"\nClassification Report:\n{classification_report(labels, predictions)}")
    for i, (pred, expected, file_path) in enumerate(zip(predictions, labels, file_paths)):
        logging.info(f"Sample {i}: Filename = {file_path} Predicted = {pred}, Expected = {expected}")


def predict_flow(training_dir: str, unknown_dir: str):
    """
    Perform model training and prediction based on user input.

    Args:
        training_dir (str): Path to the directory containing training data.
        unknown_dir (str): Path to the directory containing unknown data.
    """
    use_pca = input("Use PCA for prediction? (yes/no, default: no): ").strip().lower() or 'no'
    use_pca = use_pca == 'yes'
    n_components = int(input("Specify the number of PCA components (default: 2): ").strip() or 2) if use_pca else None

    model_choice = input("Choose a model (svc, decision_tree, random_forest, knn): ").strip().lower()
    if model_choice not in model_hyperparameters:
        print("Invalid model choice. Please choose a valid model.")
        return

    model_info = model_hyperparameters[model_choice]
    model_class = model_info['model_class']
    param_grid = model_info['param_grid']

    X, y, _ = collect_data(training_dir)
    if len(X) == 0:
        logging.warning("No training data available.")
        return

    X_unk, labels_unk, file_paths_unk = collect_data(unknown_dir, file_extension='.csv', extract_labels=True)
    if len(X_unk) == 0:
        logging.warning("No unknown data available for prediction.")
        return

    training_mode = input("Run until 100% accuracy on unknown data or specify iterations (type '100%' or 'iter', default: '100%')?: ").strip().lower() or '100%'
    if training_mode == '100%':
        max_retries = int(input("Specify the maximum number of attempts to achieve 100% accuracy (default: 20): ").strip() or 20)
        accuracy_threshold = float(input("Specify the desired accuracy threshold on unknown data (default: 100.0 for perfect accuracy): ").strip() or 100.0)
        pca, clf, best_accuracy = train_until_100(X, y, X_unk, labels_unk, model_class, param_grid, max_retries=max_retries, use_pca=use_pca, accuracy_threshold=accuracy_threshold, n_components=n_components)
    elif training_mode == 'iter':
        num_iterations = int(input("Specify the number of iterations (default: 10): ").strip() or 10)
        best_accuracy = 0.0
        best_model = None
        pca = None

        if use_pca:
            pca, X_pca = perform_pca(X, n_components=n_components)
            X_unk_pca = pca.transform(X_unk)
        else:
            X_pca = X
            X_unk_pca = X_unk

        for iteration in range(1, num_iterations + 1):
            current_model = grid_search_model(X_pca, y, model_class, param_grid, cv=5)
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

    predictions = predict_folder(pca, clf, X_unk, use_pca)
    validate_predictions(predictions, labels_unk, file_paths_unk, model_choice)

    combined_plot_choice = input("Would you like to plot the combined PCA/t-SNE projection after predictions? (yes/no, default: yes): ").strip().lower() or 'yes'
    if combined_plot_choice == 'yes':
        plot_method = input("Choose 'pca' for PCA or 'tsne' (default: 'pca'): ").strip().lower() or 'pca'
        perplexity = int(input("Specify the perplexity for t-SNE (default: 30): ").strip() or 30) if plot_method == 'tsne' else None
        plot_predictions(X, y, X_unk, predictions, file_paths_unk, method=plot_method, n_components=n_components or 2, perplexity=perplexity or 30, accuracy=best_accuracy)


def main():
    """
    Main function to handle data plotting and classification based on user input.
    """
    base_dir = os.getcwd()
    training_dir = os.path.join(base_dir, 'trData')
    unknown_dir = os.path.join(base_dir, 'unkData')

    while True:
        choice = input("Choose 'plot' for PCA/t-SNE plotting, 'predict' for prediction, or 'exit' to quit: ").strip().lower()
        if choice == 'exit':
            print("Exiting the program.")
            break
        elif choice == 'plot':
            plot_flow(training_dir)
        elif choice == 'predict':
            predict_flow(training_dir, unknown_dir)
        else:
            print("Invalid input. Please choose 'plot', 'predict', or 'exit'.")


if __name__ == '__main__':
    main()
