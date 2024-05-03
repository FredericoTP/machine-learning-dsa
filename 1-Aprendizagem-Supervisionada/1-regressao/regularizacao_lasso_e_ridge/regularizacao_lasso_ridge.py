import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, Ridge, LassoCV
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings("ignore")

# Carregando os dados
train = pd.read_csv("dados/treino.csv")
test = pd.read_csv("dados/teste.csv")

# Visualizando todos os dados
all_data = pd.concat(
    (
        train.loc[:, "MSSubClass":"SaleCondition"],
        test.loc[:, "MSSubClass":"SaleCondition"],
    )
)

print(all_data.head(10))

# Pré-Processamento dos dados
new_price = {
    "price": train["SalePrice"],
    "log(price + 1)": np.log1p(train["SalePrice"]),
}
prices = pd.DataFrame(new_price)
matplotlib.rcParams["figure.figsize"] = (8.0, 5.0)
prices.hist()
plt.savefig("histograms.png")

# Log transform da variável target e remoção dos valores NA
train["SalePrice"] = np.log1p(train["SalePrice"])
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
print(skewed_feats)

# Aplicação das transformações a todos os dados
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())

# Nova divisão em dados de treino e teste
X_train = all_data[: train.shape[0]]
X_test = all_data[train.shape[0] :]
y_train = train.SalePrice


# Função para calcular o RMSE
def rmse_cv(modelo):
    rmse = np.sqrt(
        -cross_val_score(
            modelo, X_train, y_train, scoring="neg_mean_squared_error", cv=5
        )
    )
    return rmse


# ------------------------------------------------------


# Modelo de Regressão Linear Múltipla (sem regularização)

# Criando o modelo
modelo_lr = LinearRegression(fit_intercept=True)

# Treinando o modelo com dados não padronizados (em escalas diferentes)
modelo_lr.fit(X_train, y_train)

# Erro médio do modelo
print("Erro médio modelo sem regularização:", rmse_cv(modelo_lr).mean())


# ------------------------------------------------------


# Modelo de Regressão Ridge

# Cria o modelo Ridge
modelo_ridge = Ridge()

# Cross validation para encontrar os melhores valores dos parâmetros do modelo Ridge
cross_val_score(modelo_ridge, X_train, y_train, scoring="neg_mean_squared_error", cv=5)

# Erro médio do modelo
print("Erro médio modelo Regressão Ridge:", rmse_cv(modelo_ridge).mean())


# ------------------------------------------------------


# Modelo de Regressão LASSO

# Cria o modelo LASSO
modelo_lasso = LassoCV(alphas=[1, 0.1, 0.001, 0.0005]).fit(X_train, y_train)

# Erro médio do modelo
print("Erro médio modelo Regressão LASSO:", rmse_cv(modelo_lasso).mean())

# Coeficientes LASSO
coef = pd.Series(modelo_lasso.coef_, index=X_train.columns)

# Coeficientes LASSO mais relevantes e menos relevantes para o modelo
imp_coef = pd.concat([coef.sort_values().head(10), coef.sort_values().tail(10)])
matplotlib.rcParams["figure.figsize"] = (8.0, 10.0)
imp_coef.plot(kind="barh")
plt.savefig("Coeficientes_LASSO.png")
