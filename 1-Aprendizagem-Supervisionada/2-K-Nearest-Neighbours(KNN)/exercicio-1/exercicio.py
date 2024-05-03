# O exercício consiste em desenvolver o algoritmo KNN usando apenas
# linguagem Python e Numpy sem a utilização de frameworks.

# No dataset fornecido, cada planta possui 4 variáveis preditoras
# e uma variável representando a classe.

# Seu algoritmo KNN deve prever a classe de uma nova planta.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# Carregando o dataset
names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "classe"]
iris_data = pd.read_csv("data/iris.data", names=names)
print(iris_data.head())

# Separando variáveis preditoras e variável target
X = iris_data.iloc[:, :4].values
y = iris_data.iloc[:, 4]

# Labels da variável target
target_class = pd.get_dummies(iris_data["classe"]).columns
target_names = np.array(target_class)

# Convertendo as classes para valores numéricos correspondentes
y = y.replace(target_names[0], 0)
y = y.replace(target_names[1], 1)
y = y.replace(target_names[2], 2)
y = np.array(y)

# Separando os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=33
)
print("Shape das variáveis X_train e y_train: ", X_train.shape, y_train.shape)


# Função para calcular a distância euclidiana
def distancia_euclidiana(att1, att2):
    dist = 0
    for i in range(len(att1)):
        dist += pow((att1[i] - att2[i]), 2)
    return np.sqrt(dist)


# Algoritmo KNN
def KNN(array, k):

    # Array para o resultado final
    resultado = []

    # Loop por todos os elementos do array recebido como entrada
    for i in range(len(array)):
        valor = array[i]

        # Votação
        def vote(item):
            val = []
            for i in range(len(knn)):
                temp = item[i][1]
                val.append(temp)
            class_val = max(set(val), key=val.count)
            return class_val

        # Aplicando a função de distância aos dados
        distance = []
        for j in range(len(X_train)):

            # Calcula a distância de cada ponto de dado de entrada (array) para cada ponto de dado de trino
            euclidean_distance = distancia_euclidiana(valor, X_train[j])

            # Cria uma lista contendo a distância calculada e o valor do label do dado de treino em j
            temp = [euclidean_distance, y_train[j]]

            # Adiciona o item anterior à lista de distâncias
            distance.append(temp)

        # Ordena
        distance.sort()

        # Obtém o valor de k para os vizinhos mais próximos
        knn = distance[:k]

        # Faz a votação
        resultado.append(vote(knn))
    return resultado


# Avaliando o modelo
y_test_pred = KNN(X_test, 5)
y_test_prediction = np.asarray(y_test_pred)

# Calculando a acurácia
acc = y_test - y_test_prediction
err = np.count_nonzero(acc)
acuracia = ((len(y_test) - err) / len(y_test)) * 100
print("Acurácia: ", acuracia)


# Fazendo previsões para 5 novas plantas com K igual a 3
previsoes = KNN(
    [
        [6.7, 3.1, 4.4, 1.4],
        [4.6, 3.2, 1.4, 0.2],
        [4.6, 3.2, 1.4, 0.2],
        [6.4, 3.1, 5.5, 1.8],
        [6.3, 3.2, 5.6, 1.9],
    ],
    3,
)
print("Previsões para 5 novas plantas com K igual a 3: ", previsoes)

# Fazendo previsões para 5 novas plantas com K igual a 5
previsoes = KNN(
    [
        [6.7, 3.1, 4.4, 1.4],
        [4.6, 3.2, 1.4, 0.2],
        [4.6, 3.2, 1.4, 0.2],
        [6.4, 3.1, 5.5, 1.8],
        [6.3, 3.2, 5.6, 1.9],
    ],
    3,
)
print("Previsões para 5 novas plantas com K igual a 5: ", previsoes)
