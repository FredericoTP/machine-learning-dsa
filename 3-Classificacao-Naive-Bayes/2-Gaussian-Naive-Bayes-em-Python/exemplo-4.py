import numpy as np
from random import random
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pylab as pl
import matplotlib
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

# Massa de dados representando 3 classes
leopardo_features = [(random() * 5 + 8, random() * 7 + 12) for x in range(5)]
urso_features = [(random() * 4 + 3, random() * 2 + 30) for x in range(4)]
elefante_features = [(random() * 3 + 20, (random() - 0.5) * 4 + 23) for x in range(6)]

# X
x = urso_features + elefante_features + leopardo_features

# y
y = (
    ["urso"] * len(urso_features)
    + ["elefante"] * len(elefante_features)
    + ["leopardo"] * len(leopardo_features)
)

# Plot dos dados
fig, axis = plt.subplots(1, 1)

# Classe 1
urso_weight, urso_height = zip(*urso_features)
axis.plot(urso_weight, urso_height, "ro", label="Ursos")

# Classe 2
elefante_weight, elefante_height = zip(*elefante_features)
axis.plot(elefante_weight, elefante_height, "bo", label="Elefantes")

# Classe 3
leopardo_weight, leopardo_height = zip(*leopardo_features)
axis.plot(leopardo_weight, leopardo_height, "yo", label="Leopardos")

# Eixos
axis.legend(loc=4)
axis.set_xlabel("Peso")
axis.set_ylabel("Altura")

# Plot
plt.savefig("exemplo-4-animais.png")
plt.close()


# Criando o Modelo com os dados de treino
clf = GaussianNB()
clf.fit(x, y)

# Criando dados de teste
new_xses = [[2, 3], [3, 31], [21, 23], [12, 16]]

# Previsões
print(clf.predict(new_xses))
print(clf.predict_proba(new_xses))


def plot_classification_results(clf, X, y, title):
    # Divide o dataset em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Fit dos dados com o classificador
    clf.fit(X_train, y_train)

    # Cores para o gráfico
    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
    cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

    h = 0.02  # step size in the mesh

    # Plot da fronteira de decisão
    # Usando o meshgrid do NumPy e atribuindo uma cor para cada ponto
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Previsões
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Resultados em cada cor do plot
    Z = Z.reshape(xx.shape)
    pl.figure()
    pl.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot dos pontos de dados de treino
    pl.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold)

    y_predicted = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    pl.scatter(X_test[:, 0], X_test[:, 1], c=y_predicted, alpha=0.5, cmap=cmap_bold)
    pl.xlim(xx.min(), xx.max())
    pl.ylim(yy.min(), yy.max())
    pl.title(title)
    pl.savefig("exemplo-4-multiclass-classification.png")
    pl.close()

    return score


xs = np.array(x)
ys = (
    [0] * len(urso_features)
    + [1] * len(elefante_features)
    + [2] * len(leopardo_features)
)

score = plot_classification_results(clf, xs, ys, "Multiclass Classification")
print("Classification score was: %s" % score)


# Utilizando o dataset iris
# Dataset
iris = datasets.load_iris()

# Imprimindo as 3 classes do datset
print(np.unique(iris.target))

# Split dos dados
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.4
)

# Shape
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Classificador
clf = GaussianNB()

# Resultado
score = plot_classification_results(
    clf, X_train[:, :2], y_train, "Multiclass classification"
)
print("Classification score was: %s" % score)
