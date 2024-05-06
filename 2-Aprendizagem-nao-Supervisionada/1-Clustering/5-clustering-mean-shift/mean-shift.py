# Mean Shift

# O Mean Shift é uma técnica não-paramétrica de análise de espaço de
# características para localizar os máximos de uma função de densidade.
# Pode ser usado para análise de cluster, visão computacional e processamento
# de imagem.

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets._samples_generator import make_blobs
import matplotlib.pyplot as plt
from matplotlib import style
from itertools import cycle
from mpl_toolkits.mplot3d import Axes3D

style.use("ggplot")

# Exemplo 1

# Gera massa de dados
centers = [[1, 1], [-0.75, -1], [1, -1], [-3, 2]]
X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)

# Cria o modelo

# bandwidth = Comprimento da Interação entre os exemplos,
# também conhecido como a largura de banda do algoritmo.
bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=500)

# Cria o modelo
modelo_v1 = MeanShift(bandwidth=bandwidth, bin_seeding=True)

# Treina o modelo
modelo_v1.fit(X)

# Coleta os labels, centróides e número de clusters
labels = modelo_v1.labels_
cluster_centers = modelo_v1.cluster_centers_
n_clusters_ = labels.max() + 1

# Plot
plt.figure(1)
plt.clf()
colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")

for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + ".")
    plt.plot(
        cluster_center[0],
        cluster_center[1],
        "o",
        markerfacecolor=col,
        markeredgecolor="k",
        markersize=14,
    )

    plt.title("Número Estimado de Clusters: %d" % n_clusters_)
plt.savefig("1-exemplo1.png")
plt.close()


# Exemplo 2

# Gera os dados
centers = [[1, 1], [5, 5], [3, 10]]
X, _ = make_blobs(n_samples=500, centers=centers, cluster_std=1)

# Visualiza os dados
plt.scatter(X[:, 0], X[:, 1])
plt.savefig("2-dados-gerados.png")
plt.close()

# Criação do modelo
modelo_v2 = MeanShift()

# Fit
modelo_v2.fit(X)

# Coletando labels, centróides e número de clusters
labels = modelo_v2.labels_
cluster_centers = modelo_v2.cluster_centers_
n_clusters_ = len(np.unique(labels))
print(cluster_centers)
print("Número Estimado de Clusters:", n_clusters_)

# Cores
colors = 10 * ["r.", "g.", "b.", "c.", "k.", "y.", "m."]

# Plot
for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)

plt.scatter(
    cluster_centers[:, 0],
    cluster_centers[:, 1],
    marker="x",
    color="k",
    s=150,
    linewidths=5,
    zorder=10,
)
plt.savefig("3-exemplo2.png")
plt.close()


# Plot 3d

# Centróides e massa de dados
centers = [[1, 1, 1], [5, 5, 5], [3, 10, 10]]
X, _ = make_blobs(n_samples=500, centers=centers, cluster_std=1.5)

# Modelo
modelo_v3 = MeanShift()
modelo_v3.fit(X)

# Extraindo labels, centróides e número de clusters
labels = modelo_v3.labels_
cluster_centers = modelo_v3.cluster_centers_
n_clusters_ = len(np.unique(labels))

# Cores
colors = 10 * ["r", "g", "b", "c", "k", "y", "m"]

# Plot

# Área de plotagem
fig = plt.figure()

# Gráfico 3d
ax = fig.add_subplot(111, projection="3d")

# Adiciona os pontos de dados ao gráficos
for i in range(len(X)):
    ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], marker="o")

# Adiciona os centróides ao gráfico
ax.scatter(
    cluster_centers[:, 0],
    cluster_centers[:, 1],
    cluster_centers[:, 2],
    marker="x",
    color="k",
    s=150,
    linewidths=5,
    zorder=10,
)
plt.savefig("4-plot3d.png")
plt.close()
