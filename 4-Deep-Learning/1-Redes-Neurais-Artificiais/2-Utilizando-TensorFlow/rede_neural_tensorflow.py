# Rede Neural com TensorFlow Para Classificação de Imagens de Vestuário

# Imports
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Configuração de gráficos
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc("axes", labelsize=14)
mpl.rc("xtick", labelsize=12)
mpl.rc("ytick", labelsize=12)

# Para tornar a saída deste notebook estável em todas as execuções
np.random.seed(42)
tf.random.set_seed(42)


# Carregando e Preparando os Dados

# Carrega o dataset
fashion_mnist = keras.datasets.fashion_mnist

# Extraímos os dados de treino e de teste
(X_treino_full, y_treino_full), (X_teste, y_teste) = fashion_mnist.load_data()

# Shape
print("Shape: ", X_treino_full.shape)

# Tipo de dados
print("Tipo de dados: ", X_treino_full.dtype)

# Preparação dos dados
X_valid, X_treino = X_treino_full[:5000] / 255.0, X_treino_full[5000:] / 255.0
y_valid, y_treino = y_treino_full[:5000], y_treino_full[5000:]
X_teste = X_teste / 255.0

# Plot de uma imagem
plt.imshow(X_treino[0], cmap="binary")
plt.axis("off")
plt.savefig("imagem_exemplo.png")
plt.close()

# Labels (dados de saída) de treino
print("Labels: ", y_treino)

# Nomes das classes
nomes_classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Shape
print("Shape X_valid: ", X_valid.shape)

# Shape
print("Shape X_teste: ", X_teste.shape)


# Construção do Modelo

# Modelo de Rede Neural com 2 Camadas Densas

# Cria o objeto do tipo sequência
modelo = keras.models.Sequential()

# Camada para receber os dados de entrada
modelo.add(keras.layers.Flatten(input_shape=[28, 28]))

# Primeira camada oculta com ativação relu
modelo.add(keras.layers.Dense(300, activation="relu"))

# Segunda camada oculta com ativação relu
modelo.add(keras.layers.Dense(100, activation="relu"))

# Camada de saída com ativação softmax
# Teremos uma probabilidade prevista para cada classe
modelo.add(keras.layers.Dense(10, activation="softmax"))

# Limpamos a sessão Keras
keras.backend.clear_session()

# Camadas do modelo
print("Camadas do modelo:\n", modelo.layers)

# Sumário do modelo
print("Sumário do modelo:\n", modelo.summary())

# Vamos nomear a primeira camada oculta do modelo
hidden1 = modelo.layers[1]
print(hidden1.name)

# Verificamos se a camada com novo nome existe
print(modelo.get_layer(hidden1.name) is hidden1)

# Extraímos pesos e bias da primeira camada oculta
weights, biases = hidden1.get_weights()

# Pesos que serão usados no começo do treinamento e são gerados de forma aleatória pelo Keras/TensorFlow
print("Pesos:\n", weights)

# Shape
print(weights.shape)

# Bias que serão usados no começo do treinamento
print(biases)

# Shape
print(biases.shape)

# Agora compilamos o modelo com o otimizador, função de custo e a métrica
# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD
# https://www.tensorflow.org/api_docs/python/tf/keras/losses/sparse_categorical_crossentropy
modelo.compile(
    optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Treinamento
history = modelo.fit(X_treino, y_treino, epochs=50, validation_data=(X_valid, y_valid))

# Hiperparâmetros do modelo
print("Hiperparâmetros do modelo: ", history.params)

# Aqui estão as métricas disponíveis após o treinamento (erro e acurácia)
print(history.history.keys())

# Colocamos o histórico de treinamento em um dataframe, plotamos e salvamos a figura
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.savefig("histórico_treinamento.png")
plt.close()


# Avaliando o modelo

# Avalia o modelo
print(modelo.evaluate(X_teste, y_teste))

# Vamos extrair 5 imagens de teste
X_new = X_teste[:5]

# E então prever a probabilidade de cada classe para cada imagem
y_proba = modelo.predict(X_new)

# Previsões de probabilidade
print(y_proba)

# As previsões de classes são mais fáceis de interpretar
print(y_proba.round(2))

# Vamos gravar as previsões das 5 imagens
y_pred = modelo.predict_step(X_new)
print(y_pred)

# E então extraímos os nomes das classes associados a cada previsão
print(np.array(nomes_classes)[y_pred])

# Plot
plt.figure(figsize=(8, 5))
for index, image in enumerate(X_new):
    plt.subplot(1, 5, index + 1)
    plt.imshow(image, cmap="binary", interpolation="nearest")
    plt.axis("off")
    plt.title(nomes_classes[y_teste[index]], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.savefig("plot_previsoes.png")
plt.close()
