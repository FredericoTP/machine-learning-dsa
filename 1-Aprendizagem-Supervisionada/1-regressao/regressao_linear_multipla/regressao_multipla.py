import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
import random

warnings.filterwarnings("ignore")


# Carregando o dataset
boston = pd.read_csv("dados/BostonHousing.csv")

# Convertendo o dataset em um dataframe com pandas
dataset = pd.DataFrame(boston, columns=boston.columns)

# Gerando número de observações e variáveis
observations = len(dataset)
variables = dataset.columns[:-1]

# Coletando x e y
X = dataset.iloc[:, :-1]
y = dataset["medv"].values


# Usando Múltiplos Atributos com StatsModels
Xc = sm.add_constant(X)
modelo = sm.OLS(y, Xc)
modelo_v1 = modelo.fit()
print(modelo_v1.summary())

# ---------------------------------------------------------

# Matriz de Correlação - Gerando a matriz
X = dataset.iloc[:, :-1]
matriz_corre = X.corr()
print(matriz_corre)


# Criando um Correlation Plot
def visualizes_correlation_matrix(data, hurdle=0.0):
    R = np.corrcoef(data, rowvar=0)
    R[np.where(np.abs(R) < hurdle)] = 0.0
    heatmap = plt.pcolor(R, cmap=mpl.colormaps["coolwarm"], alpha=0.8)
    heatmap.axes.set_frame_on(False)
    heatmap.axes.set_yticks(np.arange(R.shape[0]) + 0.5, minor=False)
    heatmap.axes.set_xticks(np.arange(R.shape[1]) + 0.5, minor=False)
    heatmap.axes.set_xticklabels(variables, minor=False)
    plt.xticks(rotation=90)
    heatmap.axes.set_yticklabels(variables, minor=False)
    plt.tick_params(
        axis="both", which="both", bottom="off", top="off", left="off", right="off"
    )
    plt.colorbar()
    plt.savefig("Correlation_plot.png")


# Visualizando o Plot
visualizes_correlation_matrix(X, hurdle=0.5)

# ---------------------------------------------------------

# Avaliando a Multicolinearidade

# Gerando eigenvalues e eigenvectors
corr = np.corrcoef(X, rowvar=0)
eigenvalues, eigenvectors = np.linalg.eig(corr)

print(eigenvalues)
print(eigenvectors[:, 8])
print(variables[2], variables[8], variables[9])

# ---------------------------------------------------------

# Gradiente Descendente

# Aplicando Padronização

standardization = StandardScaler()
Xst = standardization.fit_transform(X)
original_means = standardization.mean_
original_stds = standardization.scale_


# Gerando X e Y
Xst = np.column_stack((Xst, np.ones(observations)))
y = dataset["medv"].values


def random_w(p):
    return np.array([np.random.normal() for j in range(p)])


def hypothesis(X, w):
    return np.dot(X, w)


def loss(X, w, y):
    return hypothesis(X, w) - y


def squared_loss(X, w, y):
    return loss(X, w, y) ** 2


def gradient(X, w, y):
    gradients = list()
    n = float(len(y))
    for j in range(len(w)):
        gradients.append(np.sum(loss(X, w, y) * X[:, j]) / n)
    return gradients


def update(X, w, y, alpha=0.01):
    return [t - alpha * g for t, g in zip(w, gradient(X, w, y))]


def optimize(X, y, alpha=0.001, eta=10**-12, iterations=1000):
    w = random_w(X.shape[1])
    path = list()
    for k in range(iterations):
        SSL = np.sum(squared_loss(X, w, y))
        new_w = update(X, w, y, alpha=alpha)
        new_SSL = np.sum(squared_loss(X, new_w, y))
        w = new_w
        if k >= 5 and (new_SSL - SSL <= eta and new_SSL - SSL >= -eta):
            path.append(new_SSL)
            return w, path
        if k % (iterations / 20) == 0:
            path.append(new_SSL)
    return w, path


# Imprimindo o resultado
alpha = 0.01
w, path = optimize(Xst, y, alpha, eta=10**-12, iterations=20000)
print("Coeficientes finais padronizados: " + ", ".join(map(lambda x: "%0.4f" % x, w)))

# Desfazendo a Padronização
unstandardized_betas = w[:-1] / original_stds
unstandardized_bias = w[-1] - np.sum((original_means / original_stds) * w[:-1])

# Imprimindo o resultado
print("%8s: %8.4f" % ("bias", unstandardized_bias))
for beta, varname in zip(unstandardized_betas, variables):
    print("%8s: %8.4f" % (varname, beta))


# ---------------------------------------------------------


# Importância do Atributos

# Criando um modelo
modelo = linear_model.LinearRegression(fit_intercept=True)

# Treinando o modelo com dados não padronizados (em escalas diferentes)
modelo.fit(X, y)

# Imprimindo os coeficientes e as variáveis
for coef, var in sorted(
    zip(map(abs, modelo.coef_), dataset.columns[:-1]), reverse=True
):
    print("%6.3f %s" % (coef, var))

# Padronizando os dados
standardization = StandardScaler()
Stand_coef_linear_reg = make_pipeline(standardization, modelo)

# Treinando o modelo com dados padronizados (na mesma escala)
Stand_coef_linear_reg.fit(X, y)

# Imprimindo os coeficients e as variáveis
for coef, var in sorted(
    zip(map(abs, Stand_coef_linear_reg.steps[1][1].coef_), dataset.columns[:-1]),
    reverse=True,
):
    print("%6.3f %s" % (coef, var))

# ---------------------------------------------------------

# Usando o R Squared
modelo = linear_model.LinearRegression(fit_intercept=True)


def r2_est(X, y):
    return r2_score(y, modelo.fit(X, y).predict(X))


print("Coeficiente R2: %0.3f" % r2_est(X, y))


# Gera o impacto de cada atributo no R2
r2_impact = list()
for j in range(X.shape[1]):
    selection = [i for i in range(X.shape[1]) if i != j]
    r2_impact.append(
        ((r2_est(X, y) - r2_est(X.values[:, selection], y)), dataset.columns[j])
    )

for imp, varname in sorted(r2_impact, reverse=True):
    print("%6.3f %s" % (imp, varname))


# ---------------------------------------------------------

# Fazendo previsões com o Modelo

# Carregando o dataset
boston = pd.read_csv("dados/BostonHousing.csv")

# Convertendo o dataset em um dataframe com pandas
dataset = pd.DataFrame(boston, columns=boston.columns)

# Formato do Dataset
print(
    "Boston housing dataset tem {} observações com {} variáveis cada uma".format(
        *dataset.shape
    )
)

# Coletando x e y
# Usaremos como variáveis explanatórias somente as 4 variáveis mais relevantes descobertas no R Squared
X = dataset[["lstat", "rm", "dis", "ptratio"]]
y = dataset["medv"].values

# Divisão em dados de treino e de teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Cria o modelo
modelo = linear_model.LinearRegression(fit_intercept=True)

# Treina o modelo
modelo_v2 = modelo.fit(X_train, y_train)

# Calcula a métrica R@ do nosso modelo
print(r2_score(y_test, modelo_v2.fit(X_train, y_train).predict(X_test)))

# Produz a matriz com os novos dados de entrada para previsão
lstat = 5
rm = 8
dis = 6
ptratio = 19

# Lista com os valores das variáveis
dados_nova_casa = [lstat, rm, dis, ptratio]

# Reshape
Xp = np.array(dados_nova_casa).reshape(1, -1)

# Previsão
print("Taxa média de ocupação para a casa: ", modelo_v2.predict(Xp))

# ---------------------------------------------------------
