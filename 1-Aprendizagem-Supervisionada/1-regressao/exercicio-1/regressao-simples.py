# Desenvolva o código necessário para a fórmula básica da regressão
# linear simples, calculando os coeficientes.

# Use o modelo para fazer previsões.

import numpy as np
import pandas as pd

# Carregando os dados
data = pd.read_csv("dados/pesos.csv")


# Definindo as variáveis x e y
X = data["Head Size"].values
Y = data["Brain Weight"].values


# Calculando os coeficientes

# Média de X e Y
mean_x = np.mean(X)
mean_y = np.mean(Y)

# Número total de valores
n = len(X)

# Usando fórmula para calcular a e b
numerador = 0
denominador = 0

for item in range(n):
    numerador += (X[item] - mean_x) * (Y[item] - mean_y)
    denominador += (X[item] - mean_x) ** 2

b = numerador / denominador
a = mean_y - (b * mean_x)

print(a, b)


# Fazendo previsão
# y = a + bx

y = a + b * 4500
print("O peso do cérebro é:", y)
