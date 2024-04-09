# Ao lidar com dados contínuos, uma suposição típica é que os valores contínuos
# associados a cada classe são distribuídos de acordo com uma
# distribuição gaussiana (distribuição normal)

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

# Dataset
iris = datasets.load_iris()

# Classificador
clf = GaussianNB()

# Modelo
modelo = clf.fit(iris.data, iris.target)

# Previsões
y_pred = modelo.predict(iris.data)

# Imprime o resultado
print(
    "Total de Observações: %d - Total de Previsões Incorretas: %d"
    % (iris.data.shape[0], (iris.target != y_pred).sum())
)
