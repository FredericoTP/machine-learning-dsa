import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# Carrega o dataset
data = pd.read_excel("dados/credit.xls", skiprows=1)
print(data.head())

# Variável target
target = "default payment next month"
y = np.asarray(data[target])

# Variáveis preditoras
features = data.columns.drop(["ID", target])
X = np.asarray(data[features])

# Divisão de dados de treino e de teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=99
)

# Classificador
clf = ExtraTreesClassifier(n_estimators=500, random_state=99)

# Treinamento do Modelo
clf.fit(X_train, y_train)

# Score
scores = cross_val_score(clf, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1)

# Imprimindo o resultado
print(
    "ExtraTreesClassifier -> Acurácia em Treino: Média = %0.3f, Desvio Padrão = %0.3f"
    % (np.mean(scores), np.std(scores))
)

# Fazendo previsões
y_pred = clf.predict(X_test)

# Confussion Matrix
confusionMatrix = confusion_matrix(y_test, y_pred)
print("Confussion Matrix: ", confusionMatrix)

# Acurácia
print("Acurácia em Teste: ", accuracy_score(y_test, y_pred))
