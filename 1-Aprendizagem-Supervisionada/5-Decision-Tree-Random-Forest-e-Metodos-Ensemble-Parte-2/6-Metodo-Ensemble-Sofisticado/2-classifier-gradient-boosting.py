# Construindo um Classificador Gradient Boosting

from sklearn.datasets import make_hastie_10_2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# Gerando o conjunto de dados
X, y = make_hastie_10_2(n_samples=5000)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Cria o classificador
est = GradientBoostingClassifier(n_estimators=200, max_depth=3)

# Cria o modelo
est.fit(X_train, y_train)

# Previsões das classes (labels)
pred = est.predict(X_test)

# Score nos dados de teste (Acurácia)
acc = est.score(X_test, y_test)
print("Acurácia: %.4f" % acc)

# Previsão das probabilidades das classes
print("Previsão da probabilidade das classes: ", est.predict_proba(X_test)[0])

# Parâmetros mais importantes quando trabalhamos com Gradient Boosting:
# Número de árvores de regressão (n_estimators)
# Profundidade de cada árvore (max_depth)
# Loss function (loss)
