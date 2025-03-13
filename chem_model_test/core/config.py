"""
Configuration file for the analyzer. This file includes hyperparameter
grids or other global constants used across multiple modules.
"""

# Example hyperparameter dictionary used by your training routines:
model_hyperparameters = {
    'svc': {
        'model_class': 'sklearn.svm.SVC',
        'param_grid': {
            'kernel': ['linear', 'rbf'],
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        }
    },
    'decision_tree': {
        'model_class': 'sklearn.tree.DecisionTreeClassifier',
        'param_grid': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 20]
        }
    },
    'random_forest': {
        'model_class': 'sklearn.ensemble.RandomForestClassifier',
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10, 20],
            'criterion': ['gini', 'entropy']
        }
    },
    'knn': {
        'model_class': 'sklearn.neighbors.KNeighborsClassifier',
        'param_grid': {
            'n_neighbors': [3, 5, 7, 10],
            'weights': ['uniform', 'distance']
        }
    }
}
