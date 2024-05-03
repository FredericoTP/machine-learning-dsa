import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Carregando o dataset
digitos = datasets.load_digits()

# Visualizando algumas imagens e labels
images_e_labels = list(zip(digitos.images, digitos.target))
for index, (image, label) in enumerate(images_e_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis("off")
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    plt.title("Label: %i" % label)

plt.savefig("images_e_labels.png")
plt.close()

# Gera X e Y
X = digitos.data
Y = digitos.target

# Formato de X e Y
print("Formato de X e Y: ", X.shape, Y.shape)


# Pré-Processamento e Normalização

# Divisão em dados de treino e de teste
X_treino, testeData, Y_treino, testeLabels = train_test_split(
    X, Y, test_size=0.30, random_state=101
)

# Divisão dos dados de treino em dados de treino e dados de validação
treinoData, validData, treinoLabels, validLabels = train_test_split(
    X_treino, Y_treino, test_size=0.1, random_state=84
)

# Imprimindo o número de exemplos (observações) em cada dataset
print("Exemplos de Treino: {}".format(len(treinoLabels)))
print("Exemplos de Validação: {}".format(len(validLabels)))
print("Exemplos de Teste: {}".format(len(testeLabels)))

# Normalização dos dados pela Média

# Cálculo da média do dataset de treino
X_norm = np.mean(X, axis=0)

# Normalização dos dados de treino e de teste
X_treino_norm = treinoData - X_norm
X_valid_norm = validData - X_norm
X_teste_norm = testeData - X_norm

# Shape dos datasets
print("Shape datasets: ", X_treino_norm.shape, X_valid_norm.shape, X_teste_norm.shape)


# Testando o melhor valor de K
# Range de valores de k que iremos testar
kVals = range(1, 30, 2)

# Lista vazia para receber as acurácias
acuracias = []

# Loop em todos os valores de k para testar cada um deles
for k in kVals:
    # Treinando o modelo KNN com cada valor de k
    modeloKNN = KNeighborsClassifier(n_neighbors=k)
    modeloKNN.fit(treinoData, treinoLabels)

    # Avaliando o modelo e atualizando a lista de acurácias
    score = modeloKNN.score(validData, validLabels)
    print("Com valor de k = %d, a acurácia é = %.2f%%" % (k, score * 100))
    acuracias.append(score)

# Obtendo o valor de K que paresentou a maior acurácia
i = np.argmax(acuracias)
print(
    "O valor de k = %d alcançou a mais alta acurácia de %.2f%% nos dados de validação!"
    % (kVals[i], acuracias[i] * 100)
)


# Construção e treinamento do Modelo KNN
# Criando a versão final do modelo com o maior valor de k
modeloFinal = KNeighborsClassifier(n_neighbors=kVals[i])

# Treinamento do modelo
modeloFinal.fit(treinoData, treinoLabels)


# Previsões com Dados de Teste e Avaliação do Modelo
# Previsões com os dados de teste
predictions = modeloFinal.predict(testeData)

# Performance do modelo nos dados de teste
print("Avaliação do Modelo nos Dados de Teste")
print(classification_report(testeLabels, predictions))

# Confussion Matrix do Modelo Final
print("Confussion Matrix")
print(confusion_matrix(testeLabels, predictions))

# Fazendo previsões com o modelo treinado usando dados de teste
for i in np.random.randint(0, high=len(testeLabels), size=(5,)):
    # Obtém uma imagem e faz a previsão
    image = testeData[i]
    prediction = modeloFinal.predict([image])[0]

    # Mostra as previsões
    imgdata = np.array(image, dtype="float")
    pixels = imgdata.reshape((8, 8))
    plt.imshow(pixels, cmap="gray")
    plt.annotate(prediction, (3, 3), bbox={"facecolor": "white"}, fontsize=16)
    print("Eu acredito que esse digito seja: {}".format(prediction))

    # Salva a imagem com um nome único
    nome_arquivo = "imagem_{}.png".format(i)
    plt.savefig(nome_arquivo)
    plt.close()


# Previsões em Novos Dados com o Modelo Treinado
# Definindo um novo digito (dados de entrada)
novoDigito = [
    0.0,
    0.0,
    0.0,
    8.0,
    15.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    12.0,
    14.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    3.0,
    16.0,
    7.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    6.0,
    16.0,
    2.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    7.0,
    16.0,
    16.0,
    13.0,
    5.0,
    0.0,
    0.0,
    0.0,
    15.0,
    16.0,
    9.0,
    9.0,
    14.0,
    0.0,
    0.0,
    0.0,
    3.0,
    14.0,
    9.0,
    2.0,
    16.0,
    2.0,
    0.0,
    0.0,
    0.0,
    7.0,
    15.0,
    16.0,
    11.0,
    0.0,
]

# Normalizando o novo digito
novoDigito_norm = novoDigito - X_norm

# Fazendo a previsão com o modelo treinado
novaPrevisao = modeloFinal.predict([novoDigito_norm])

# previsão do Modelo
imgdata = np.array(novoDigito, dtype="float")
pixels = imgdata.reshape((8, 8))
plt.imshow(pixels, cmap="gray")
plt.annotate(prediction, (3, 3), bbox={"facecolor": "white"}, fontsize=16)
print("Eu acredito que esse digito seja: {}".format(prediction))
plt.savefig("previsao.png")
plt.close()
