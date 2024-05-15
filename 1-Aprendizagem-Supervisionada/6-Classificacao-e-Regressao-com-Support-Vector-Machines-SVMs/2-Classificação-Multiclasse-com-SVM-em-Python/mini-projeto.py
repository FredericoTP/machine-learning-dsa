# Usando SVM Para Prever a Intenção de Compra de Usuários de E-Commerce

# Imports
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)
from sklearn import svm
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore")

# Carregando os dados
df_original = pd.read_csv("dados/online_shoppers_intention.csv")
print("Dataframe:\n", df_original.head())


# ------------------------------------------------------


# Análise Exploratória

# Shape
print("Shape dos dados: ", df_original.shape)

# Tipos de Dados
print("Tipos de dados:\n", df_original.dtypes)

# Verificando valores missing
print("Valores missing: ", df_original.isna().sum())

# Removendo as linhas com valores missing
df_original.dropna(inplace=True)

# Verificando valores missing
print("Valores missing: ", df_original.isna().sum())

# Shape
print("Shape dos dados: ", df_original.shape)

# Verificando Valores Únicos
print("Valores únicos:\n", df_original.nunique())

# Para fins de visualização, dividiremos os dados em variáveis contínuas
# e categóricas. Trataremos todas as variáveis com menos de 30 entradas
# únicas como categóricas.

# Preparando os dados para o plot

# Cria uma cópia do dataset original
df = df_original.copy()

# Listas vazias para os resultados
continuous = []
categorical = []

# Loop pelas colunas
for c in df.columns[:-1]:
    if df.nunique()[c] >= 30:
        continuous.append(c)
    else:
        categorical.append(c)

# Variáveis contínuas
print("Variáveis contínuas:\n", df[continuous].head())

# Variáveis categóricas
print("Variáveis categóricas:\n", df[categorical].head())

# Plot das variáveis contínuas

# Tamanho da área de plotagem
fig = plt.figure(figsize=(12, 8))

# Loop pelas variáveis contínuas
for i, col in enumerate(continuous):
    plt.subplot(3, 3, i + 1)
    df.boxplot(col)
    plt.tight_layout()

plt.savefig("imagens/boxplot1.png")
plt.close()

# Transformação de log nas variáveis contínuas
df[continuous] = np.log1p(1 + df[continuous])

# Plot das variáveis contínuas

# Tamanho da área de plotagem
fig = plt.figure(figsize=(12, 8))

# Loop pelas variáveis contínuas
for i, col in enumerate(continuous):
    plt.subplot(3, 3, i + 1)
    df.boxplot(col)
    plt.tight_layout()
plt.savefig("imagens/boxplot2.png")
plt.close()

# Matriz de Correlação Entre Variáveis Contínuas.

# Área de plotagem
plt.figure(figsize=(10, 10))

# Matriz de Correlação
sns.heatmap(
    df[
        [
            "Administrative_Duration",
            "Informational_Duration",
            "ProductRelated_Duration",
            "BounceRates",
            "ExitRates",
            "PageValues",
            "Revenue",
        ]
    ].corr(),
    vmax=1.0,
    square=True,
)

plt.savefig("imagens/matriz-de-correlacao.png")
plt.close()

# Visualização de gráficos de variáveis categóricas para analisar
# como a variável de destino é influenciada por elas.

# Countplot Venda ou Não
plt.subplot(1, 2, 2)
plt.title("Venda ou Não")
sns.countplot(df["Revenue"])
plt.savefig("imagens/countplot1.png")
plt.close()

# Countplot Tipo de Visitante
plt.xlabel("Tipo de Visitante")
sns.countplot(df["VisitorType"])
plt.savefig("imagens/countplot2.png")
plt.close()

# Stacked Bar Tipo de Visitante x Revenue
pd.crosstab(df["VisitorType"], df["Revenue"]).plot(
    kind="bar", stacked=True, figsize=(15, 5), color=["red", "green"]
)
plt.savefig("imagens/stacked1.png")
plt.close()

# Pie Chart Tipos de Visitantes
labels = ["Visitante_Retornando", "Novo_Visitante", "Outro"]
plt.title("Tipos de Visitantes")
plt.pie(df["VisitorType"].value_counts(), labels=labels, autopct="%.2f%%")
plt.legend()
plt.savefig("imagens/piechart1.png")
plt.close()

# Countplot Final de Semana ou Não
plt.subplot(1, 2, 1)
plt.title("Final de Semana ou Não")
sns.countplot(df["Weekend"])
plt.savefig("imagens/countplot3.png")
plt.close()

# Stacked Bar Final de Semana x Revenue
pd.crosstab(df["Weekend"], df["Revenue"]).plot(
    kind="bar", stacked=True, figsize=(15, 5), color=["red", "green"]
)
plt.savefig("imagens/stacked2.png")
plt.close()

# Countplot Tipos de Sistemas Operacionais
# plt.figure(figsize = (15,6))
plt.title("Tipos de Sistemas Operacionais")
plt.xlabel("Sistema Operacional Usado")
sns.countplot(df["OperatingSystems"])
plt.savefig("imagens/countplot4.png")
plt.close()

# Stacked Bar Tipo de SO x Revenue
pd.crosstab(df["OperatingSystems"], df["Revenue"]).plot(
    kind="bar", stacked=True, figsize=(15, 5), color=["red", "green"]
)
plt.savefig("imagens/stacked3.png")
plt.close()

# Countplot Tipo de Tráfego
plt.title("Tipos de Tráfego")
plt.xlabel("Tipo de Tráfego")
sns.countplot(df["TrafficType"])
plt.savefig("imagens/countplot5.png")
plt.close()

# Stacked Bar Tipos de Tráfego x Revenue
pd.crosstab(df["TrafficType"], df["Revenue"]).plot(
    kind="bar", stacked=True, figsize=(15, 5), color=["red", "green"]
)
plt.savefig("imagens/stacked4.png")
plt.close()


# ------------------------------------------------------


# Pré-Processamento dos Dados

# Dataset
print("Dataset:\n", df_original.head())

# Cria o encoder
lb = LabelEncoder()

# Aplica o encoder nas variáveis que estão com string
df_original["Month"] = lb.fit_transform(df_original["Month"])
df_original["VisitorType"] = lb.fit_transform(df_original["VisitorType"])

# Remove valores missing eventualmente gerados
df_original.dropna(inplace=True)

# Dataset
print("Dataset:\n", df_original.head())

# Shape
print("Shape: ", df_original.shape)

# Verificando se a variável resposta está balanceada
target_count = df_original.Revenue.value_counts()
print("Verificando balanceamento: ", target_count)

# Plot
sns.countplot(df_original.Revenue, palette="OrRd")
plt.box(False)
plt.xlabel("Receita (Revenue) Por Sessão Não (0) / Sim (1)", fontsize=11)
plt.ylabel("Total Sessões", fontsize=11)
plt.title("Contagem de Classes\n")
plt.savefig("imagens/balanceamento1.png")
plt.close()

# Variáveis explicativas
print("Variáveis explicativas:\n", df_original.iloc[:, 0:17].head())

# Variável Target
print("Variável Target:\n", df_original.iloc[:, 17].head())

# Seed para reproduzir o mesmo resultado
seed = 100

# Separa X e y
X = df_original.iloc[:, 0:17]
y = df_original.iloc[:, 17]

# Cria o balanceador SMOTE
smote_bal = SMOTE(random_state=seed)

# Aplica o balanceador
X_res, y_res = smote_bal.fit_resample(X, y)

# Plot
sns.countplot(y_res, palette="OrRd")
plt.box(False)
plt.xlabel("Receita (Revenue) Por Sessão Não (0) / Sim (1)", fontsize=11)
plt.ylabel("Total Sessões", fontsize=11)
plt.title("Contagem de Classes\n")
plt.savefig("imagens/balanceamento2.png")
plt.close()

# Shape dos dados originais
print("Shape dados originais: ", df_original.shape)

# Shape dos dados reamostrados
print("Shape dados reamostrados X: ", X_res.shape)

# Shape dos dados reamostrados
print("Shape dados reamostrados y: ", y_res.shape)

# Ajustando X e y
X = X_res
y = y_res

# Divisão em Dados de Treino e Teste.
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# ------------------------------------------------------


# Modelo SVM

# Modelo base com Kernel Linear

# Cria o modelo
modelo_v1 = svm.SVC(kernel="linear")

# Treinamento
start = time.time()
modelo_v1.fit(X_treino, y_treino)
end = time.time()
print("Tempo de Treinamento do Modelo:", end - start)

# Previsões
previsoes_v1 = modelo_v1.predict(X_teste)

# Dicionário de métricas e metadados
SVM_dict_v1 = {
    "Modelo": "SVM",
    "Versão": "1",
    "Kernel": "Linear",
    "Precision": precision_score(previsoes_v1, y_teste),
    "Recall": recall_score(previsoes_v1, y_teste),
    "F1 Score": f1_score(previsoes_v1, y_teste),
    "Acurácia": accuracy_score(previsoes_v1, y_teste),
    "AUC": roc_auc_score(y_teste, previsoes_v1),
}

# Print
print("Métricas em Teste:\n", SVM_dict_v1)


# Modelo com Kernel Linear e Dados Padronizados (Scaled)

# Padronização
sc = StandardScaler()
X_treino_scaled = sc.fit_transform(X_treino)
X_teste_scaled = sc.transform(X_teste)

# Cria o modelo
modelo_v2 = svm.SVC(kernel="linear")

# Treinamento
start = time.time()
modelo_v2.fit(X_treino_scaled, y_treino)
end = time.time()
print("Tempo de Treinamento do Modelo:", end - start)

# Previsões
previsoes_v2 = modelo_v2.predict(X_teste_scaled)

# Dicionário de métricas e metadados
SVM_dict_v2 = {
    "Modelo": "SVM",
    "Versão": "2",
    "Kernel": "Linear com Dados Padronizados",
    "Precision": precision_score(previsoes_v2, y_teste),
    "Recall": recall_score(previsoes_v2, y_teste),
    "F1 Score": f1_score(previsoes_v2, y_teste),
    "Acurácia": accuracy_score(previsoes_v2, y_teste),
    "AUC": roc_auc_score(y_teste, previsoes_v2),
}

# Print
print("Métricas em Teste:\n", SVM_dict_v2)


# Otimização de Hiperparâmetros com Grid Search e Kernel RBF

# Cria o modelo
modelo_v3 = svm.SVC(kernel="rbf")

# Valores para o grid
C_range = np.array([50.0, 100.0, 200.0])
gamma_range = np.array([0.3 * 0.001, 0.001, 3 * 0.001])

# Grid de hiperparâmetros
svm_param_grid = dict(gamma=gamma_range, C=C_range)

# Grid Search
start = time.time()
modelo_v3_grid_search_rbf = GridSearchCV(modelo_v3, svm_param_grid, cv=3)

# Treinamento
modelo_v3_grid_search_rbf.fit(X_treino_scaled, y_treino)
end = time.time()
print("Tempo de Treinamento do Modelo com Grid Search:", end - start)

# Acurácia em Treino
print(f"Acurácia em Treinamento: {modelo_v3_grid_search_rbf.best_score_ :.2%}")
print("")
print(f"Hiperparâmetros Ideais: {modelo_v3_grid_search_rbf.best_params_}")

# Previsões
previsoes_v3 = modelo_v3_grid_search_rbf.predict(X_teste_scaled)

# Dicionário de métricas e metadados
SVM_dict_v3 = {
    "Modelo": "SVM",
    "Versão": "3",
    "Kernel": "RBF com Dados Padronizados",
    "Precision": precision_score(previsoes_v3, y_teste),
    "Recall": recall_score(previsoes_v3, y_teste),
    "F1 Score": f1_score(previsoes_v3, y_teste),
    "Acurácia": accuracy_score(previsoes_v3, y_teste),
    "AUC": roc_auc_score(y_teste, previsoes_v3),
}

# Print
print("Métricas em Teste:\n", SVM_dict_v3)


# Otimização de Hiperparâmetros com Grid Search e Kernel Polinomial

# Cria o modelo
modelo_v4 = svm.SVC(kernel="poly")

# Valores para o grid
r_range = np.array([0.5, 1])
gamma_range = np.array([0.001, 0.01])
d_range = np.array([2, 3, 4])

# Grid de hiperparâmetros
param_grid_poly = dict(gamma=gamma_range, degree=d_range, coef0=r_range)

# Grid Search
start = time.time()
modelo_v4_grid_search_poly = GridSearchCV(modelo_v4, param_grid_poly, cv=3)

# Treinamento
modelo_v4_grid_search_poly.fit(X_treino_scaled, y_treino)
end = time.time()
print("Tempo de Treinamento do Modelo com Grid Search:", end - start)

# Acurácia em Treino
print(f"Acurácia em Treinamento: {modelo_v4_grid_search_poly.best_score_ :.2%}")
print("")
print(f"Hiperparâmetros Ideais: {modelo_v4_grid_search_poly.best_params_}")

# Previsões
previsoes_v4 = modelo_v4_grid_search_poly.predict(X_teste_scaled)

# Dicionário de métricas e metadados
SVM_dict_v4 = {
    "Modelo": "SVM",
    "Versão": "4",
    "Kernel": "Polinomial com Dados Padronizados",
    "Precision": precision_score(previsoes_v4, y_teste),
    "Recall": recall_score(previsoes_v4, y_teste),
    "F1 Score": f1_score(previsoes_v4, y_teste),
    "Acurácia": accuracy_score(previsoes_v4, y_teste),
    "AUC": roc_auc_score(y_teste, previsoes_v4),
}

# Print
print("Métricas em Teste:\n", SVM_dict_v4)


# Concatena todos os dicionários em um dataframe do Pandas
resumo = pd.DataFrame(
    {
        "SVM_dict_v1": pd.Series(SVM_dict_v1),
        "SVM_dict_v2": pd.Series(SVM_dict_v2),
        "SVM_dict_v3": pd.Series(SVM_dict_v3),
        "SVM_dict_v4": pd.Series(SVM_dict_v4),
    }
)

# Print
print("Resumo:\n", resumo)


# ------------------------------------------------------


# Fazendo Previsões com o Modelo Treinado

# Novo registro
novo_x = np.array(
    [4.0, 5.56, 1.0, 3.78, 2.995, 6.00, 0.69, 0.70, 0.69, 0, 6, 1, 1, 3, 3, 2, False]
).reshape(1, -1)

# Padronizando o registro
novo_x_scaled = StandardScaler().fit_transform(novo_x)

# Previsão
previsao_novo_x = modelo_v3_grid_search_rbf.predict(novo_x_scaled)

print("Previsão: ", previsao_novo_x)
