import numpy as np
from time import time
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_digits

# Obtém o dataset
digits = load_digits()
X, y = digits.data, digits.target

# Construindo o classificador
clf = RandomForestClassifier(n_estimators=20)

# ----------------------------

# Randomized Search

# Valores dos parâmetros que serão testados
param_dist = {
    "max_depth": [3, None],
    "max_features": sp_randint(1, 11),
    "min_samples_leaf": sp_randint(1, 11),
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy"],
}

# Executando o Randomized Search
n_iter_search = 20
random_search = RandomizedSearchCV(
    clf, param_distributions=param_dist, n_iter=n_iter_search, return_train_score=True
)

start = time()
random_search.fit(X, y)
print(
    "RandomizedSearchCV executou em %.2f segundos para %d candidatos a parâmetros do modelo."
    % ((time() - start), n_iter_search)
)

# Imprime as combinaçoes dos parâmetros e suas respectivas médias de acurácia
print(random_search.cv_results_)

# ----------------------------

# Grid Search

# Usando um grid completo de todos os parâmetros
param_grid = {
    "max_depth": [3, None],
    "max_features": [1, 3, 10],
    "min_samples_leaf": [1, 3, 10],
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy"],
}

# Executando o Grid Search
grid_search = GridSearchCV(clf, param_grid=param_grid, return_train_score=True)

start = time()
grid_search.fit(X, y)
print(
    "GridSearchCV executou em %.2f segundos para todas as combinações de candidatos a parâmetros do modelo."
    % (time() - start)
)

print(grid_search.cv_results_)
