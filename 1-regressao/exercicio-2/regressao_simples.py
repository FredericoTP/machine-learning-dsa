import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
from scipy.stats import pearsonr
import statsmodels.api as sm

warnings.filterwarnings("ignore")


# Carregando o Dataset
boston = pd.read_csv("dados/BostonHousing.csv")


# Convertendo o Dataset em um dataframe com Pandas
dataset = pd.DataFrame(boston, columns=boston.columns)


# Calculando a média da variável de resposta
valor_medio_esperado_na_previsão = dataset["medv"].mean()
print("Valor médio esperado na previsão:", valor_medio_esperado_na_previsão)


# Calculando (simulando o SSE)
# O SSE é a diferença ao quadrado entre o valor previsto e o valor observado.
# Considerando que o valor previsto seja igual a média, podemos considerar que
# y = média da variável target (valores observados).
# Estamos apenas simulando o SSE, uma vez que a regressão ainda não foi criada e os valores previstos ainda não foram calculados.

squared_erros = pd.Series(valor_medio_esperado_na_previsão - dataset["medv"]) ** 2
SSE = np.sum(squared_erros)
print("Soma dos Quadrados dos Erros (SSE): %01.f" % SSE)


# Histograma dos erros
# Temos mais error "pequenos", ou seja, mais valores próximos à média.
hist_plot = squared_erros.plot(kind="hist")
plt.savefig("histograma.png")


# Função para calcular o desvio padrão
def calc_desvio_padrao(variable, bias=0):
    observations = float(len(variable))
    return np.sqrt(
        np.sum((variable - np.mean(variable)) ** 2) / (observations - min(bias, 1))
    )


# Imprimindo o desvio padrão via fórmula e via Numpy da variável RM
print(
    "Resultado da Função: %0.5f Resultado do Numpy: %0.5f"
    % (calc_desvio_padrao(dataset["rm"]), np.std(dataset["rm"]))
)


# Funções para calcular a variância da variável RM e a correlação com a variável target
def covariance(variable_1, variable_2, bias=0):
    observations = float(len(variable_1))
    return np.sum(
        (variable_1 - np.mean(variable_1)) * (variable_2 - np.mean(variable_2))
    ) / (observations - min(bias, 1))


def standardize(variable):
    return (variable - np.mean(variable)) / np.std(variable)


def correlation(var1, var2, bias=0):
    return covariance(standardize(var1), standardize(var2), bias)


# Compara o resultado das nossas funções com a função pearsonr do SciPy
print(
    "Nossa estimativa de Correlação: %0.5f"
    % (correlation(dataset["rm"], dataset["medv"]))
)
print(
    "Correlação a partir da função pearsonr do SciPy: %0.5f"
    % pearsonr(dataset["rm"], dataset["medv"])[0]
)


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


# Criando o Modelo com StatsModels


# Gerando X e Y. Vamos adicionar a constante ao valor de X, gerando uma matrix.
Y = dataset["medv"]
X = dataset["rm"]


# Esse comando adiciona os valores dos coeficientes à variável X (o bias será calculado internamente pela função)
X = sm.add_constant(X)


# Criando o modelo de regressão
modelo = sm.OLS(Y, X)


# Treinando o modelo
modelo_vl = modelo.fit()


print(modelo_vl.summary())
print(modelo_vl.params)


# Gerando valores previstos
valores_previstos = modelo_vl.predict(X)


# Fazendo previsões com o modelo treinado
RM = 5
Xp = np.array([1, RM])
print(
    "Se RM = %0.1f nosso modelo prevê que a mediana da taxa de ocupação é %0.1f"
    % (RM, modelo_vl.predict(Xp))
)


# Gerando um ScatterPlot com a linha de Regressão
# Definindo o range dos valores de x e y
x_range = [dataset["rm"].min(), dataset["rm"].max()]
y_range = [dataset["medv"].min(), dataset["medv"].max()]


# Primeira camada do Scatter Plot
scatter_plot = dataset.plot(
    kind="scatter", x="rm", y="medv", xlim=x_range, ylim=y_range
)


# Segunda camada do Scatter Plot
meanY = scatter_plot.plot(x_range, [dataset["medv"].mean(), dataset["medv"].mean()])
meanX = scatter_plot.plot([dataset["rm"].mean(), dataset["rm"].mean()], y_range)
plt.savefig("2.png")


# Terceira camada do Scatter Plot
regression_line = scatter_plot.plot(dataset["rm"], valores_previstos)
plt.savefig("3.png")


# Gerando Resíduos
residuos = dataset["medv"] - valores_previstos
residuos_normalizados = standardize(residuos)


# Scatter Plot dos resíduos
residual_scatter_plot = plt.plot(dataset["rm"], residuos_normalizados, "bp")
plt.xlabel("RM")
plt.ylabel("Resíduos Normalizados")
mean_residual = plt.plot([int(x_range[0]), round(x_range[1], 0)], [0, 0])
upper_bound = plt.plot([int(x_range[0]), round(x_range[1], 0)], [3, 3])
lower_bound = plt.plot([int(x_range[0]), round(x_range[1], 0)], [-3, -3])
plt.grid()
plt.savefig("4.png")
