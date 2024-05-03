import numpy as np
from sklearn.naive_bayes import GaussianNB
from astroML.plotting import setup_text_plots
from matplotlib import pyplot as plt
from matplotlib import colors


setup_text_plots(fontsize=8, usetex=True)

# Criando massa de dados
np.random.seed(0)
mu1 = [1, 1]
cov1 = 0.3 * np.eye(2)
mu2 = [5, 3]
cov2 = np.eye(2) * np.array([0.4, 0.1])

# Concatenando
X = np.concatenate(
    [
        np.random.multivariate_normal(mu1, cov1, 100),
        np.random.multivariate_normal(mu2, cov2, 100),
    ]
)

y = np.zeros(200)
y[100:] = 1

# Criação do Modelo
clf = GaussianNB()
clf.fit(X, y)

# Previsões
xlim = (-1, 8)
ylim = (-1, 5)
xx, yy = np.meshgrid(
    np.linspace(xlim[0], xlim[1], 71), np.linspace(ylim[0], ylim[1], 81)
)
Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z[:, 1].reshape(xx.shape)

# Plot dos resultados
fig = plt.figure(figsize=(5, 3.75))
ax = fig.add_subplot(111)
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.binary, zorder=2)

ax.contour(xx, yy, Z, [0.5], colors="k")

ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

plt.savefig("exemplo-3.png")
plt.close()
