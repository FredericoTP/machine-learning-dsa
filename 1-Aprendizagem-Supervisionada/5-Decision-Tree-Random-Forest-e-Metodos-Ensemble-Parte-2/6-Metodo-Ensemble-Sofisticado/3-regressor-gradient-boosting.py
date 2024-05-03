import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from matplotlib import pyplot as plt
from itertools import islice
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

# Construindo um Regressor Gradient Boosting

FIGSIZE = (11, 7)


# Aproximação da função (linha de regressão ideal)
def reg_line(x):
    return x * np.sin(x) + np.sin(2 * x)


# Gerando dados de treino e de teste
def gen_data(n_samples=200):

    # Gera a massa de dados aleatórios
    np.random.seed(15)
    X = np.random.uniform(0, 10, size=n_samples)[:, np.newaxis]
    y = reg_line(X.ravel()) + np.random.normal(scale=2, size=n_samples)

    # Divide em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=3
    )

    return X_train, X_test, y_train, y_test


# Construção dos datasets
X_train, X_test, y_train, y_test = gen_data(100)

# Dados para a linha de regressão
x_plot = np.linspace(0, 10, 500)


# Plot dos dados
def plot_data(name, alpha=0.4, s=20):

    # Cria a figura
    fig = plt.figure(figsize=FIGSIZE)

    # Gera o plot
    gt = plt.plot(x_plot, reg_line(x_plot), alpha=alpha)

    # Plot dos dados de treino e teste
    plt.scatter(X_train, y_train, s=s, alpha=alpha)
    plt.scatter(X_test, y_test, s=s, alpha=alpha)
    plt.xlim((0, 10))
    plt.ylabel("y")
    plt.xlabel("x")
    plt.savefig(name)


# Formatação
annotation_kw = {
    "xycoords": "data",
    "textcoords": "data",
    "arrowprops": {"arrowstyle": "->", "connectionstyle": "arc"},
}

# Plot
plot_data("dados-treino-teste.png")
# Azul - Treino
# Vermelho - Teste

# --------------------------------------

# Plot de 2 Árvores com diferentes profundidades

# Árvore de decisão com max_depth = 1
est = DecisionTreeRegressor(max_depth=1).fit(X_train, y_train)
plt.plot(
    x_plot,
    est.predict(x_plot[:, np.newaxis]),
    label="max_depth=1",
    color="g",
    alpha=0.9,
    linewidth=3,
)
plt.savefig("arvore-max-depth-1.png")

# Árvore de decisão com max_depth = 3
est = DecisionTreeRegressor(max_depth=3).fit(X_train, y_train)
plt.plot(
    x_plot,
    est.predict(x_plot[:, np.newaxis]),
    label="max_depth=3",
    color="g",
    alpha=0.7,
    linewidth=1,
)
plt.savefig("arvore-max-depth-3.png")
plt.close()

# --------------------------------------

# Aplicando o Gradient Boosting Regressor

plot_data("dados-treino-teste.png")

# Regressor GBRT
est = GradientBoostingRegressor(n_estimators=1000, max_depth=1, learning_rate=1.0)

# Modelo
est.fit(X_train, y_train)
ax = plt.gca()
first = True

# Passos através das previsões à medida que adicionamos mais árvores
for pred in islice(est.staged_predict(x_plot[:, np.newaxis]), 0, est.n_estimators_, 10):
    plt.plot(x_plot, pred, color="r", alpha=0.2)
    if first:
        ax.annotate(
            "Alto Viés - Baixa Variância",
            xy=(x_plot[x_plot.shape[0] // 2], pred[x_plot.shape[0] // 2]),
            xytext=(4, 4),
            **annotation_kw
        )
        first = False

# Previsões
pred = est.predict(x_plot[:, np.newaxis])
plt.plot(x_plot, pred, color="r", label="GBRT max_depth=1")
ax.annotate(
    "Baixo Viés - Alta Variância",
    xy=(x_plot[x_plot.shape[0] // 2], pred[x_plot.shape[0] // 2]),
    xytext=(6.25, -6),
    **annotation_kw
)

plt.savefig("gradient-boosting-regressor.png")
plt.close()

# --------------------------------------

# Diagnosticando se o modelo sofre de Overfitting


def deviance_plot(
    est,
    X_test,
    y_test,
    ax=None,
    label="",
    train_color="#2c7bb6",
    test_color="#d7191c",
    alpha=1.0,
    ylim=(0, 10),
):
    n_estimators = len(est.estimators_)
    test_dev = np.empty(n_estimators)

    for i, pred in enumerate(est.staged_predict(X_test)):
        test_dev[i] = np.mean((pred - y_test) ** 2)

    if ax is None:
        fig = plt.figure(figsize=FIGSIZE)
        ax = plt.gca()

    ax.plot(
        np.arange(n_estimators) + 1,
        test_dev,
        color=test_color,
        label="Teste %s" % label,
        linewidth=2,
        alpha=alpha,
    )
    ax.plot(
        np.arange(n_estimators) + 1,
        est.train_score_,
        color=train_color,
        label="Treino %s" % label,
        linewidth=2,
        alpha=alpha,
    )
    ax.set_ylabel("Erro")
    ax.set_xlabel("Número de Estimadores Base")
    ax.set_ylim(ylim)
    return test_dev, ax


# Aplica a função aos dados de teste para medir o overfitting do nosso modelo (est)
test_dev, ax = deviance_plot(est, X_test, y_test)
ax.legend(loc="upper right")

# Legendas
ax.annotate(
    "Menor nível de erro no dataset de Teste",
    xy=(test_dev.argmin() + 1, test_dev.min() + 0.02),
    xytext=(150, 3.5),
    **annotation_kw
)

ann = ax.annotate(
    "",
    xy=(800, test_dev[799]),
    xycoords="data",
    xytext=(800, est.train_score_[799]),
    textcoords="data",
    arrowprops={"arrowstyle": "<->"},
)
ax.text(810, 3.5, "Gap Treino-Teste")

plt.savefig("verifica-overfitting-1.png")

# --------------------------------------

# Regularização (Evitar Overfitting)
# 1- Alterar a estrutura da árvore
# 2- Shrinkage
# 3- Stochastic Gradient Boosting

# Alterando a Estrutura da Árvore
# Alterando o parâmetro min_samples_leaf farantimos um número maior
# de amostras por folha.


def fmt_params(params):
    return ", ".join("{0}={1}".format(key, val) for key, val in params.items())


fig = plt.figure(figsize=FIGSIZE)
ax = plt.gca()

for params, (test_color, train_color) in [
    ({}, ("#d7191c", "#2c7bb6")),
    ({"min_samples_leaf": 3}, ("#fdae61", "#abd9e9")),
]:
    est = GradientBoostingRegressor(n_estimators=1000, max_depth=1, learning_rate=1.0)
    est.set_params(**params)
    est.fit(X_train, y_train)
    test_dev, ax = deviance_plot(
        est,
        X_test,
        y_test,
        ax=ax,
        label=fmt_params(params),
        train_color=train_color,
        test_color=test_color,
    )

ax.annotate(
    "Alto Viés", xy=(900, est.train_score_[899]), xytext=(600, 3), **annotation_kw
)
ax.annotate(
    "Baixa Variância", xy=(900, test_dev[899]), xytext=(600, 3.5), **annotation_kw
)
plt.legend(loc="upper right")
plt.savefig("verifica-overfitting-2.png")


# Shrinkage
# Reduz o aprendizado de cada árvore reduzindo a learning_rate.

fig = plt.figure(figsize=FIGSIZE)
ax = plt.gca()

for params, (test_color, train_color) in [
    ({}, ("#d7191c", "#2c7bb6")),
    ({"learning_rate": 0.1}, ("#fdae61", "#abd9e9")),
]:
    est = GradientBoostingRegressor(n_estimators=1000, max_depth=1, learning_rate=1.0)
    est.set_params(**params)
    est.fit(X_train, y_train)

    test_dev, ax = deviance_plot(
        est,
        X_test,
        y_test,
        ax=ax,
        label=fmt_params(params),
        train_color=train_color,
        test_color=test_color,
    )

ax.annotate(
    "Requer mais árvores",
    xy=(200, est.train_score_[199]),
    xytext=(300, 1.75),
    **annotation_kw
)
ax.annotate(
    "Menor erro no dataset de teste",
    xy=(900, test_dev[899]),
    xytext=(600, 1.75),
    **annotation_kw
)

plt.legend(loc="upper right")
plt.savefig("verifica-overfitting-3.png")


# Stochastic Gradient Boosting
# Cria subsamples do dataset de treino antes de crescer cada árvore.
# Cria subsamples dos atributos antes de encontrar o melhor split node (max_features). Funciona melhor se houver grande volume de dados.

fig = plt.figure(figsize=FIGSIZE)
ax = plt.gca()
for params, (test_color, train_color) in [
    ({}, ("#d7191c", "#2c7bb6")),
    ({"learning_rate": 0.1, "subsample": 0.7}, ("#fdae61", "#abd9e9")),
]:
    est = GradientBoostingRegressor(
        n_estimators=1000, max_depth=1, learning_rate=1.0, random_state=1
    )
    est.set_params(**params)
    est.fit(X_train, y_train)
    test_dev, ax = deviance_plot(
        est,
        X_test,
        y_test,
        ax=ax,
        label=fmt_params(params),
        train_color=train_color,
        test_color=test_color,
    )

ax.annotate(
    "Menor Taxa de Erro no Dataset de Teste",
    xy=(400, test_dev[399]),
    xytext=(500, 3.0),
    **annotation_kw
)

plt.legend(loc="upper right", fontsize="small")
plt.savefig("verifica-overfitting-4.png")
plt.close()

# --------------------------------------

# Tunning dos Hiperparâmetros com Grid Search

# Grid de parâmetros
param_grid = {
    "learning_rate": [0.1, 0.01, 0.001],
    "max_depth": [4, 5, 6],
    "min_samples_leaf": [3, 4, 5],
    "subsample": [0.3, 0.5, 0.7],
    "n_estimators": [400, 700, 1000, 2000, 3000],
}

# Regressor
est = GradientBoostingRegressor()

# Modelo criado com GridSearchCV
gs_cv = GridSearchCV(est, param_grid, scoring="neg_mean_squared_error", n_jobs=4).fit(
    X_train, y_train
)

# Imprime os melhors parâmetros
print("Melhores Hiperparâmetros: %r" % gs_cv.best_params_)

# Recria o modelo com os melhores parâmetros

est.set_params(**gs_cv.best_params_)
est.fit(X_train, y_train)

# Plot
plot_data("dados-treino-teste.png")
plt.plot(x_plot, est.predict(x_plot[:, np.newaxis]), color="r", linewidth=2)
plt.savefig("tunning.png")
plt.close()
