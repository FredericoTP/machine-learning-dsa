# Mini-Projeto - Usando a Rede Neural Para Prever a Ocorrência de Câncer

# Imports
import sklearn
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from construindo_rede_neural import modeloNN, predict


# Carregamos o objeto completo
temp = load_breast_cancer()

# Tipo do objeto
print("Tipo do objeto: ", type(temp))

# Visualiza o objeto
print("Objeto:\n", temp)

# Carregamos o dataset
dados = pd.DataFrame(
    columns=load_breast_cancer()["feature_names"], data=load_breast_cancer()["data"]
)

# Shape
print("Shape: ", dados.shape)

# Visualiza os dados
print("Dados:\n", dados.head())

# Verifica se temos valores ausentes
print("Valores missing:\n", dados.isnull().any())

# Separa a variável target
target = load_breast_cancer()["target"]

# Visualiza a variável
print("Variável Target:\n", target)

# Total de registros por classe - Câncer Benigno
print("Câncer Benigno: ", np.count_nonzero(target == 1))

# Total de registros por classe - Câncer Maligno
print("Câncer Maligno: ", np.count_nonzero(target == 0))

# Vamos extrair os labels

# Dicionário para os labels
labels = {}

# Nomes das classes da variável target
target_names = load_breast_cancer()["target_names"]

# Mapeamento
for i in range(len(target_names)):
    labels.update({i: target_names[i]})

# Visualiza os labels
print("Labels: ", labels)

# Agora preparamos as variáveis preditoras em X
X = np.array(dados)

# Visualiza os dados de entrada
print("Dados de entrada:\n", X)

# Dividimos os dados de entrada e saída em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, target, test_size=0.15, shuffle=True
)

# Shape dos dados de treino
print(X_treino.shape)
print(y_treino.shape)

# Shape dos dados de teste
print(X_teste.shape)
print(y_teste.shape)

# Ajusta o shape dos dados de entrada
X_treino = X_treino.T
X_teste = X_teste.T
print(X_treino.shape)
print(X_teste.shape)

# Precisamos ajustar também os dados de saída
y_treino = y_treino.reshape(1, len(y_treino))
y_teste = y_teste.reshape(1, len(y_teste))
print(y_treino.shape)
print(y_teste.shape)

# Variável com as dimensões de entrada para oo número de neurônios
dims_camada_entrada = [X_treino.shape[0], 50, 20, 5, 1]
print(dims_camada_entrada)

# Treinamento do modelo

print("\nIniciando o Treinamento.\n")

parametros, custo = modeloNN(
    X=X_treino,
    Y=y_treino,
    dims_camada_entrada=dims_camada_entrada,
    num_iterations=3000,
    learning_rate=0.0075,
)

print("\nTreinamento Concluído.\n")

# Plot do erro durante o treinamento
plt.plot(custo)
plt.savefig("erro-durante-treinamento.png")
plt.close()

# Previsões com os dados de treino
y_pred_treino = predict(X_treino, parametros)

# Visualiza as previsões
print("Previsões:\n", y_pred_treino)

# Ajustamos o shape em treino
y_pred_treino = y_pred_treino.reshape(-1)
y_treino = y_treino.reshape(-1)

print(y_pred_treino > 0.5)

# Convertemos as previsões para o valor binário de classe
# (0 ou 1, usando como threshold o valor de 0.5 da probabilidade)
y_pred_treino = 1 * (y_pred_treino > 0.5)

print(y_pred_treino)

# Calculamos a acurácia comparando valor real com valor previsto
acc_treino = sum(1 * (y_pred_treino == y_treino)) / len(y_pred_treino) * 100
print("Acurácia nos dados de treino: " + str(acc_treino))

print(
    classification_report(y_treino, y_pred_treino, target_names=["Maligno", "Benigno"])
)

# Previsões com o modelo usando dados de teste
y_pred_teste = predict(X_teste, parametros)

# Visualiza os dados
print(y_pred_teste)

# Ajustamos os shapes
y_pred_teste = y_pred_teste.reshape(-1)
y_teste = y_teste.reshape(-1)

# Convertemos as previsões para o valor binário de classe
y_pred_teste = 1 * (y_pred_teste > 0.5)

# Visualizamos as previsões
print("Previsões: ", y_pred_teste)

# Calculamos a acurácia
acuracia = sum(1 * (y_pred_teste == y_teste)) / len(y_pred_teste) * 100
print("Acurácia nos dados de teste: " + str(acuracia))
print(classification_report(y_teste, y_pred_teste, target_names=["Maligno", "Benigno"]))
