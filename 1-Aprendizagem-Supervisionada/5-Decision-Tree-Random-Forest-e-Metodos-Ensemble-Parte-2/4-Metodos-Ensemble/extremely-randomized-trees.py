# Bagging - Extremely Randomized Trees

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score

# Carregando os dados
digits = load_digits()

# Pré-processamento
data = scale(digits.data)

# Varáveis preditoras e variável target
X = data
y = digits.target

# Cria o classificador com uma árvore de decisão
clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y)
mean = scores.mean()
print("Scores DecisionTree: ", scores)
print("Média dos scores: ", mean)

# Cria o classificador com Random Forest
clf = RandomForestClassifier(
    n_estimators=10, max_depth=None, min_samples_split=2, random_state=0
)
scores = cross_val_score(clf, X, y)
mean = scores.mean()
print("Scores RandomForest: ", scores)
print("Média dos scores: ", mean)

# Cria o classificador com Extra Tree
clf = ExtraTreesClassifier(
    n_estimators=10, max_depth=None, min_samples_split=2, random_state=0
)
scores = cross_val_score(clf, X, y)
mean = scores.mean()
print("Scores ExtraTree: ", scores)
print("Média dos scores: ", mean)
