import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Carregando o dataset
credito_dataset = pd.read_csv("dados/credit_dataset_final.csv")

# Pré-processamento


# Transformando variáveis em fatores
def to_factors(df, variables):
    for variable in variables:
        df[variable] = df[variable].astype("category")
    return df


# Normalização
def scale_features(df, variables):
    for variable in variables:
        df[variable] = (df[variable] - df[variable].mean()) / df[variable].std()
    return df


# Normalizando as variáveis
numeric_vars = ["credit.duration.months", "age", "credit.amount"]
credito_dataset_scaled = scale_features(credito_dataset, numeric_vars)

# Variáveis do tipo fator
categorical_vars = [
    "credit.rating",
    "account.balance",
    "previous.credit.payment.status",
    "credit.purpose",
    "savings",
    "employment.duration",
    "installment.rate",
    "marital.status",
    "guarantor",
    "residence.duration",
    "current.assets",
    "other.credits",
    "apartment.type",
    "bank.credits",
    "occupation",
    "dependents",
    "telephone",
    "foreign.worker",
]

# Aplicando as conversões ao dataset
credito_dataset_final = to_factors(credito_dataset_scaled, categorical_vars)

# Preparando os dados de treino e de teste
train_data, test_data = train_test_split(
    credito_dataset_final, test_size=0.4, random_state=42
)

# Separando os atributos e as classes
train_features = train_data.iloc[:, 1:]
train_target = train_data.iloc[:, 0]
test_features = test_data.iloc[:, 1:]
test_target = test_data.iloc[:, 0]

# Construindo o modelo de regressão logística
formula_init = "credit.rating ~ " + " + ".join(train_features.columns)
modelo_v1 = LogisticRegression()
modelo_v1.fit(train_features, train_target)

# Visualizando os detalhes do modelo
print(modelo_v1.coef_)
print(modelo_v1.intercept_)

# Fazendo previsões e analisando o resultado
previsoes = modelo_v1.predict(test_features)

# Confusion Matrix
conf_matrix = confusion_matrix(test_target, previsoes)
print(conf_matrix)

# Feature Selection
# (Note: Feature selection is not directly translated, as scikit-learn's LogisticRegression does not provide variable importance)
# You might want to use other techniques or libraries for feature selection in Python.

# Construindo um novo modelo com as variáveis selecionadas
selected_features = [
    "account.balance",
    "credit.purpose",
    "previous.credit.payment.status",
    "savings",
    "credit.duration.months",
]
formula_new = "credit.rating ~ " + " + ".join(selected_features)
modelo_v2 = LogisticRegression()
modelo_v2.fit(train_data[selected_features], train_data["credit.rating"])

# Prevendo e avaliando o modelo
previsoes_new = modelo_v2.predict(test_data[selected_features])
conf_matrix_new = confusion_matrix(test_data["credit.rating"], previsoes_new)
print(conf_matrix_new)

# Avaliando a performance do modelo
# ROC Curve
fpr, tpr, thresholds = roc_curve(
    test_data["credit.rating"],
    modelo_v2.predict_proba(test_data[selected_features])[:, 1],
)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure()
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=2,
    label="ROC curve (area = {:.2f})".format(roc_auc),
)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")

# Fazendo previsões em novos dados
novo_dataset = pd.DataFrame(
    {
        "account.balance": [1, 3, 3, 2],
        "credit.purpose": [4, 2, 3, 2],
        "previous.credit.payment.status": [3, 3, 2, 2],
        "savings": [2, 3, 2, 3],
        "credit.duration.months": [15, 12, 8, 6],
    }
)

# Aplica as transformações
novo_dataset = to_factors(
    novo_dataset,
    ["account.balance", "credit.purpose", "previous.credit.payment.status", "savings"],
)
novo_dataset = scale_features(novo_dataset, ["credit.duration.months"])

# Previsões
previsoes_novo_client = modelo_v2.predict(novo_dataset[selected_features])
print(previsoes_novo_client)
