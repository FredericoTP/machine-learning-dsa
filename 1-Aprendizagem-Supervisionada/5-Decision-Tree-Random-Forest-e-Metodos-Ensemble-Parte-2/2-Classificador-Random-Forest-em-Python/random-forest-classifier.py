import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# Criando Random Forest Classifier - 1

# Gera o dataset
digitos = load_digits()

# Aplica Escala nos dados
data = scale(digitos.data)
print("data: ", data)
print("data.shape: ", data.shape)

# Obtém número de observações e número de atributos
n_observ, n_features = data.shape
print("n_observ: ", n_observ)
print("n_features: ", n_features)

# Obtém os labels
n_digits = len(np.unique(digitos.target))
labels = digitos.target
print("labels: ", labels)

# Cria o classificador
clf = RandomForestClassifier(n_estimators=10)

# Os 4 principais parâmetros em Modelos de random Forest são:
# n_estimators -> quanto maior, melhor! (número de árvores a serem criadas)
# max_depth -> o padrão é none e nesse caso árvores completas são criadas. Ajustando esse parâmetro pode ajudar a evitar overfitting.
# max_features -> diferentes valores devem ser testados, pois este parâmetro impacta na forma como os modelos RF distribuem os atributos pelas árvores.
# criterion -> define a forma como o algoritmo fará a divisão dos atributos e a classificação dos nós em cada árvore.

# Construção do modelo
clf = clf.fit(data, labels)

scores = clf.score(data, labels)
print("Score: ", scores)

# Extraindo a importância
importances = clf.feature_importances_
indices = np.argsort(importances)

# Obtém os índices
ind = []
for i in indices:
    ind.append(labels[i])

# Plot da Importância dos Atributos
plt.figure(1)
plt.title("Importância dos Atributos")
plt.barh(range(len(indices)), importances[indices], color="b", align="center")
plt.yticks(range(len(indices)), ind)
plt.xlabel("Importância Relativa")
plt.savefig("importancia-relativa-rf-1.png")
plt.close()

# ---------------------------------------------------------

# Criando Random Forest Classifier - 2

from treeinterpreter import treeinterpreter as ti
from sklearn.datasets import load_iris

# Carrega o dataset
iris = load_iris()

# Cria o classificador
rf = RandomForestClassifier(max_depth=4)

# Obtém os índices a partir do comprimento da variável target
idx = list(range(len(iris.target)))

# Randomiza o índice
np.random.shuffle(idx)

# Cria o modelo
rf.fit(iris.data[idx][:100], iris.target[idx][:100])

# Obtém as instâncias (exemplos ou observações) e retorna as probabilidades
instance = iris.data[idx][100:101]
print(rf.predict_proba(instance))

prediction, bias, contributions = ti.predict(rf, instance)
print("Previsões: ", prediction)
print("Contribuição dos atributos:")
for item, feature in zip(contributions[0], iris.feature_names):
    print(feature, item)
