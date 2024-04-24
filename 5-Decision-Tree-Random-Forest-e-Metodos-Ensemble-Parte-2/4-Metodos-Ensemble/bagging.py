# Bagging

# Bagging é usado para construção de múltiplos modelos (normalmente do
# mesmo tipo) a partir de diferentes subsets no dataset de treino.

# Um classificador Bagging é um meta-estimador ensemble que faz fit de
# classificadores base.

from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# Carga de dados
digits = load_digits()

plt.gray()
plt.matshow(digits.images[5])
plt.savefig("digits.png")
plt.close()

# Pré-processamento
# Coloca todos os dados na mesma escala
data = scale(digits.data)

# Variáveis preditoras e variável target
X = data
y = digits.target

# Construção do Classificador
bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)

# Score do modelo
scores = cross_val_score(bagging, X, y)

# Média do score
mean = scores.mean()

print("Scores: ", scores)
print("Média do score: ", mean)
