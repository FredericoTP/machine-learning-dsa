# Gaussian Naive Bayes - Exemplo 2

from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

# Dataset
dataset = datasets.load_iris()

# Classificador
clf = GaussianNB()

# Modelo
modelo = clf.fit(dataset.data, dataset.target)
print(modelo)

# Previsões
observado = dataset.target
previsto = modelo.predict(dataset.data)

# Sumário
print(metrics.classification_report(observado, previsto))
print(metrics.confusion_matrix(observado, previsto))
