import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import warnings

warnings.filterwarnings("ignore")

# Carregando o dataset Smarket
smarket = pd.read_csv(
    "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/ISLR/Smarket.csv"
)
smarket["Direction"] = pd.Categorical(
    smarket["Direction"]
).codes  # Convertendo para valores numéricos

# Divisão em dados de treino e teste
X = smarket.drop("Direction", axis=1)
y = smarket["Direction"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=300
)

# Normalização dos dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Testando diferentes valores de k
k_range = list(range(1, 30, 2))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=10, scoring="accuracy")
    k_scores.append(scores.mean())

# Obtendo o melhor valor de k
best_k = k_range[np.argmax(k_scores)]
print("O melhor valor de k é:", best_k)

# Construção do Modelo KNN com o melhor valor de k
modelo_final = KNeighborsClassifier(n_neighbors=best_k)
modelo_final.fit(X_train_scaled, y_train)

# Avaliação do modelo
print("Avaliação do Modelo nos Dados de Teste:")
print(classification_report(y_test, modelo_final.predict(X_test_scaled)))
print("Confusion Matrix:")
print(confusion_matrix(y_test, modelo_final.predict(X_test_scaled)))
