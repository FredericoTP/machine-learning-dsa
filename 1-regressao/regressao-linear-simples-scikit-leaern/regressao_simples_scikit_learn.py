import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn import linear_model
import warnings

warnings.filterwarnings("ignore")


# Carregando o dataset
boston = pd.read_csv("dados/BostonHousing.csv")

# Convertendo o dataset em um dataframe com pandas
dataset = pd.DataFrame(boston, columns=boston.columns)

# Calculando a média da variável de resposta
media_previsao = dataset["medv"].mean()
print("Valor médio esperado na previsão:", media_previsao)

# Calculando (simulando o SSE)
# O SSE é a diferença ao quadrado entre o valor previsto e o valor observado.
# Considerando que o valor previsto seja igual a média, podemos considerar que
# y = média da variável target (valores observados).
# Estamos apenas simulando o SSE, uma vez que a regressão ainda não foi criada e os valores previstos ainda não foram calculados.

squared_erros = pd.Series(media_previsao - dataset["medv"]) ** 2
SSE = np.sum(squared_erros)
print("Soma dos Quadrados dos Erros (SSE): %01.f" % SSE)

# Histograma dos erros
# Temos mais error "pequenos", ou seja, mais valores próximos à média.
hist_plot = squared_erros.plot(kind="hist")
plt.savefig("histograma.png")

# Calculando o desvio padrão
calc_desvio_padrao = np.std(dataset["rm"])
print("Desvio padrão: %0.5f" % calc_desvio_padrao)

# Calculando a correlação de RM com a variável target
correlation = pearsonr(dataset["rm"], dataset["medv"])[0]
print("Correlação de RM com a variável target: %0.5f" % correlation)

# Definindo o range dos valores de x e y
x_range = [dataset["rm"].min(), dataset["rm"].max()]
y_range = [dataset["medv"].min(), dataset["medv"].max()]

# Plot dos valores de x e y com a média
scatter_plot = dataset.plot(
    kind="scatter", x="rm", y="medv", xlim=x_range, ylim=y_range
)

meanY = scatter_plot.plot(x_range, [dataset["medv"].mean(), dataset["medv"].mean()])
meanX = scatter_plot.plot([dataset["rm"].mean(), dataset["rm"].mean()], y_range)
plt.savefig("2.png")


# Criando o modelo com o Scikit-Learn

# Cria o objeto
modelo_v2 = linear_model.LinearRegression(fit_intercept=True)

# Define os valores de x e y
num_observ = len(dataset)
X = dataset["rm"].values.reshape(
    (num_observ, 1)
)  # X deve sempre ser uma matriz e nunca um vetor
Y = dataset["medv"].values  # Y pode ser um vetor

# Treinando o modelo - fit()
modelo_v2.fit(X, Y)

# Imprime os coeficientes
print(modelo_v2.coef_)
print(modelo_v2.intercept_)

# Imprime as previsões
print(modelo_v2.predict(X))

# Fazendo previsões com o modelo treinado
RM = 5
Xp = np.array(RM).reshape(-1, 1)
print(
    "Se RM = %0.1f nosso modelo prevê que a mediana da taxa de ocupação é %0.1f"
    % (RM, modelo_v2.predict(Xp))
)


# Minimizando a Cost Function com Pseudo-Inversão

# Definindo x e y
num_observ = len(dataset)
X = dataset["rm"].values.reshape(
    (num_observ, 1)
)  # X deve sempre ser uma matriz e nunca um vetor
Xb = np.column_stack((X, np.ones(num_observ)))
Y = dataset["medv"].values  # Y pode ser um vetor


# Funções para matriz inversa e equações normais
def matriz_inversa(X, Y, pseudo=False):
    if pseudo:
        return np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))
    else:
        return np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))


def normal_equations(X, Y):
    return np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))


# Imprime os valores
print(matriz_inversa(Xb, Y))
print(matriz_inversa(Xb, Y, pseudo=True))
print(normal_equations(Xb, Y))
