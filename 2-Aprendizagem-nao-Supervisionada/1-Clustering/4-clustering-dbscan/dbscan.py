# DBSCAN

# DBSCAN (Density-Based Spatial Clustering of Applications with Noise) é um
# algoritmo de clustering popular usado como uma alternativa ao K-Means,
# em análise preditiva. Ele não requer que você defina o número de clusters.
# Mas em troca, você tem que ajustar dois outros parâmetros.

# Imports
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons, load_iris
from sklearn.decomposition import PCA

# Gerando os dados
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
plt.scatter(X[:, 0], X[:, 1])
plt.tight_layout()
plt.savefig("1-dados-gerados.png")
plt.close()

# Construção do modelo
modelo = DBSCAN(eps=0.2, min_samples=5, metric="euclidean")

# Fit do modelo
y_db = modelo.fit_predict(X)

# Plot
plt.scatter(
    X[y_db == 0, 0], X[y_db == 0, 1], c="blue", marker="o", s=40, label="Cluster 1"
)
plt.scatter(
    X[y_db == 1, 0], X[y_db == 1, 1], c="red", marker="s", s=40, label="Cluster 2"
)
plt.legend()
plt.tight_layout()
plt.savefig("2-dados-clusterizados.png")
plt.close()


# DBSCAN com Dataset Iris

# Carregando os dados
iris = load_iris()

# Primeira versão do modelo
dbscan_v1 = DBSCAN()

# Fit
dbscan_v1.fit(iris.data)

# Labels
print("Labels: ", dbscan_v1.labels_)

# Reduzindo a Dimensionalidade
pca = PCA(n_components=2).fit(iris.data)

# Fit
pca_2d = pca.transform(iris.data)

# Plot
for i in range(0, pca_2d.shape[0]):
    if dbscan_v1.labels_[i] == 0:
        c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c="r", marker="+")
    elif dbscan_v1.labels_[i] == 1:
        c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c="g", marker="o")
    elif dbscan_v1.labels_[i] == -1:
        c3 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c="b", marker="*")

plt.legend([c1, c2, c3], ["Cluster 1", "Cluster 2", "Noise"])
plt.title("DBSCAN Gerou 2 Clusters e Encontrou Noise")
plt.savefig("3-dbscan-iris-v1.png")
plt.close()

# Segunda versão do modelo
dbscan_v2 = DBSCAN(eps=0.8, min_samples=4, metric="euclidean")
dbscan_v2.fit(iris.data)

# Plot
for i in range(0, pca_2d.shape[0]):
    if dbscan_v2.labels_[i] == 0:
        c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c="r", marker="+")
    elif dbscan_v2.labels_[i] == 1:
        c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c="g", marker="o")
    elif dbscan_v2.labels_[i] == -1:
        c3 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c="b", marker="*")

plt.legend([c1, c2, c3], ["Cluster 1", "Cluster 2", "Noise"])
plt.title("DBSCAN Gerou 2 Clusters e Encontrou Noise")
plt.savefig("4-dbscan-iris-v2.png")
plt.close()
