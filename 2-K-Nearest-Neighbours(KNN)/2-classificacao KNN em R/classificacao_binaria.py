# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import confusion_matrix
# import warnings

# warnings.filterwarnings("ignore")

# # Carregando o dataset
# Smarket = pd.read_csv(
#     "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/ISLR/Smarket.csv"
# )
# print(Smarket)

# # Gera X e Y
# X = Smarket.drop(columns=["Direction"])
# Y = Smarket["Direction"]

# # Divisão em dados de treino e de teste
# X_treino, testeData, Y_treino, testelabels = train_test_split(
#     X, Y, test_size=0.30, random_state=300
# )

# # Divisão dos dados de treino em dados de treino e dados de validação
# treinoData, validData, treinoLabels, validLabels = train_test_split(
#     X_treino, Y_treino, test_size=0.1, random_state=84
# )

# # Normalização dos dados
# scaler = StandardScaler()
# X_treino_norm = scaler.fit_transform(treinoData)
# X_valid_norm = scaler.transform(validData)
# X_teste_norm = scaler.transform(testeData)

# # Shape dos datasets
# print("Shape datasets: ", X_treino_norm.shape, X_valid_norm.shape, X_teste_norm.shape)

# # Testando o melhor valor de K
# # Range de valores de k que iremos testar
# kVals = range(1, 150, 2)

# # Lista vazia para receber as acurácias
# acuracias = []

# # Loop em todos os valores de k para testar cada um deles
# for k in kVals:
#     # Treinando o modelo KNN com cada valor de k
#     modeloKNN = KNeighborsClassifier(n_neighbors=k)
#     modeloKNN.fit(treinoData, treinoLabels)

#     # Avaliando o modelo e atualizando a lista de acurácias
#     score = modeloKNN.score(X_valid_norm, validLabels)
#     print("Com valor de k = %d, a acurácia é = %.2f%%" % (k, score * 100))
#     acuracias.append(score)

# # Obtendo o valor de K que paresentou a maior acurácia
# i = np.argmax(acuracias)
# print(
#     "O valor de k = %d alcançou a mais alta acurácia de %.2f%% nos dados de validação!"
#     % (kVals[i], acuracias[i] * 100)
# )

# # Construção e treinamento do Modelo KNN
# # Criando a versão final do modelo com o maior valor de k
# modeloFinal = KNeighborsClassifier(n_neighbors=kVals[i])

# # Treinamento do modelo
# modeloFinal.fit(treinoData, treinoLabels)

# # Previsões com Dados de Teste e Avaliação do Modelo
# # Previsões com os dados de teste
# predictions = modeloFinal.predict(testeData)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Carregando o dataset Smarket
smarket = pd.read_csv(
    "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/ISLR/Smarket.csv"
)
smarket["Direction"] = pd.Categorical(
    smarket["Direction"]
).codes  # Convertendo para valores numéricos

# Divisão em dados de treino e teste
X = smarket.drop("Direction", axis=1)
y = smarket["Direction"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=300
)

# Normalização dos dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Testando diferentes valores de k
k_range = list(range(1, 30, 2))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=10, scoring="accuracy")
    k_scores.append(scores.mean())

# Obtendo o melhor valor de k
best_k = k_range[np.argmax(k_scores)]
print("O melhor valor de k é:", best_k)

# Construção do Modelo KNN com o melhor valor de k
modelo_final = KNeighborsClassifier(n_neighbors=best_k)
modelo_final.fit(X_train_scaled, y_train)

# Avaliação do modelo
print("Avaliação do Modelo nos Dados de Teste:")
print(classification_report(y_test, modelo_final.predict(X_test_scaled)))
print("Confusion Matrix:")
print(confusion_matrix(y_test, modelo_final.predict(X_test_scaled)))
