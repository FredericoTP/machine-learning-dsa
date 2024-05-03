# Um classificador AdaBoost é um meta-estimador que começa ajustando
# um classifcador no conjunto de dados original e depois ajusta cópias
# adicionais do classificador no mesmo conjunto de dados, mas onde os
# pesos das instâncias classificadas incorretamente são ajustadas para
# que os classificadores subsequentes se concentrem mais em casos difíceis.

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score

# Carregando os dados
heart = fetch_openml("heart")

# Variáveis preditoras e variável target
X = heart.data
y = np.copy(heart.target)
y[y == -1] = 0

print(X.shape)
print(y)

# Datasets de treino e de teste
X_test, y_test = X[189:], y[189:]
X_train, y_train = X[:189], y[:189]

# Construindo o estimador base
estim_base = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)

# Construindo a primeira versão do modelo Adaboost
ada_clf_v1 = AdaBoostClassifier(
    estimator=estim_base, learning_rate=0.1, n_estimators=400, algorithm="SAMME"
)

# Treinamento do modelo
ada_clf_v1.fit(X_train, y_train)

# Score
scores = cross_val_score(ada_clf_v1, X_test, y_test)
means = scores.mean()
print("Scores: ", scores)
print("Média: ", means)

# Construindo a segunda versão do modelo Adaboost
ada_clf_v2 = AdaBoostClassifier(
    estimator=estim_base, learning_rate=1.0, n_estimators=400, algorithm="SAMME"
)

# Treinamento do modelo
ada_clf_v2.fit(X_train, y_train)

# Score
scores = cross_val_score(ada_clf_v2, X_test, y_test)
means = scores.mean()
print("Scores: ", scores)
print("Média: ", means)
