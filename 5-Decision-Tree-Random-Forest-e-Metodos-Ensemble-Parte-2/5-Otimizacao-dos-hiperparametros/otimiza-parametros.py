import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
    RandomizedSearchCV,
)
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# Carrega o dataset
data = pd.read_excel("dados/credit.xls", skiprows=1)
print(data.head())

# Variável target
target = "default payment next month"
y = np.asarray(data[target])

# Variáveis preditoras
features = data.columns.drop(["ID", target])
X = np.asarray(data[features])

# Divisão de dados de treino e de teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=99
)

# Classificador
clf = ExtraTreesClassifier(n_estimators=500, random_state=99)

# Treinamento do Modelo
clf.fit(X_train, y_train)

# Score
scores = cross_val_score(clf, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1)

# Imprimindo o resultado
print(
    "ExtraTreesClassifier -> Acurácia em Treino: Média = %0.3f, Desvio Padrão = %0.3f"
    % (np.mean(scores), np.std(scores))
)

# Fazendo previsões
y_pred = clf.predict(X_test)

# Confussion Matrix
confusionMatrix = confusion_matrix(y_test, y_pred)
print("Confussion Matrix: ", confusionMatrix)

# Acurácia
print("Acurácia em Teste: ", accuracy_score(y_test, y_pred))

# -------------------------------------------

# Otimização dos Hiperparâmetros com Randomized Search

# O Randomized Search gera amostras dos parâmetros dos algoritmos a partir de
# uma distribuição randômica uniforme para um número fixo de iterações. Um
# modelo é construído e testado para cada combinação de parâmetros.

# Definição dos parâmetros
param_dist = {
    "max_depth": [1, 3, 7, 8, 12, None],
    "max_features": [8, 9, 10, 11, 16, 22],
    "min_samples_split": [8, 10, 11, 14, 16, 19],
    "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7],
    "bootstrap": [True, False],
}

# Para o classificador criado com ExtraTrees, testamos diferentes combinações de parâmetros
rsearch = RandomizedSearchCV(
    clf, param_distributions=param_dist, n_iter=25, return_train_score=True
)

# Aplicando o resultado ao conjunto de dados de treino e obtendo o score
rsearch.fit(X_train, y_train)

# Resultados
print("Resultados: ", rsearch.cv_results_)

# Imprimindo o melhor estimador
bestclf = rsearch.best_estimator_
print("Melhor estimador: ", bestclf)

# Aplicando o melhor estimador para realizar as previsões
y_pred = bestclf.predict(X_test)

# Confussion Matrix
confusionMatrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ", confusionMatrix)

# Acurácia
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia: ", accuracy)
