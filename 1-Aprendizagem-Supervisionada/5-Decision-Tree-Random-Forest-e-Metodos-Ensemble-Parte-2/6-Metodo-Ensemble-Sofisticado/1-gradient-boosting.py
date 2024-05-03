import numpy as np
from sklearn.datasets import make_hastie_10_2, make_friedman1
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


# Gradient Boosting Classifier

# Define dados para x e y
X, y = make_hastie_10_2(random_state=0)
X_train, X_test = X[:2000], X[2000:]
y_train, y_test = y[:2000], y[2000:]

# Cria o classificador
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, random_state=0)

# Treina o classificador
clf.fit(X_train, y_train)

# Calcula a acurácia (score)
print("Score: ", clf.score(X_test, y_test))

# -------------------------------------------------------

# Gradient Boosting Regressor

# Define dados para x e y
X, y = make_friedman1(n_samples=1200, random_state=0, noise=1.0)
X_train, X_test = X[:200], X[200:]
y_train, y_test = y[:200], y[200:]

# Cria o regressor
est = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=1,
    random_state=0,
    loss="squared_error",
)

# Treina o regressor
est.fit(X_train, y_train)

# Calcula o erro médio quadrado
print("Erro médio quadrado: ", mean_squared_error(y_test, est.predict(X_test)))
