# Identificando os Clusters

# Imports
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pylab as pl

# Carregando o dataset
iris = load_iris()
print("Dataset: ", iris)

# Visualizando o tipo de objeto dos dados
print("Tipo do objeto de dados: ", type(iris.data))

# Visualizando as 20 primeiras linhas
print(iris.data[1:20,])

# Criando o modelo
kmeans = KMeans(n_clusters=3, random_state=111)
kmeans.fit(iris.data)
print("Labels: ", kmeans.labels_)


# Criando o Cluster Map
cluster_map = pd.DataFrame(iris.data)
cluster_map["cluster"] = kmeans.labels_
print(cluster_map)

# Filtrando os Dados Pelo Cluster
print(cluster_map[cluster_map.cluster == 2])


# Reduzindo a dimensionalidade
pca = PCA(n_components=2).fit(iris.data)

# Aplicando o PCA
pca_2d = pca.transform(iris.data)
print("pca_2d: ", pca_2d)
print("Shape: ", pca_2d.shape)

# Gerando "labels" para os resultados dos clusters
for i in range(0, pca_2d.shape[0]):

    if kmeans.labels_[i] == 0:
        c1 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c="r", marker="+")

    elif kmeans.labels_[i] == 1:
        c2 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c="g", marker="o")

    elif kmeans.labels_[i] == 2:
        c3 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c="b", marker="*")
        pl.legend([c1, c2, c3], ["Cluster 0", "Cluster 1", "Cluster 2"])
        pl.title("Clusters K-means com Iris dataset em 3 clusters")
pl.savefig("clusters-kmeans.png")
pl.close()
