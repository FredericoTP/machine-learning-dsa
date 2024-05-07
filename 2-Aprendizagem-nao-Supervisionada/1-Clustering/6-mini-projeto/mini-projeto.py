# Mini-Projeto

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings("ignore")

# Carregando os dados
dataset = pd.read_csv(
    "dados/household_power_consumption.txt", delimiter=";", low_memory=False
)
print("Dados:\n", dataset.head())
print("Shape: ", dataset.shape)
print("Tipos dos dados:\n", dataset.dtypes)

# Checando se há valores missing
missing_values = dataset.isnull().values.any()
print("Valores missing: ", missing_values)

# Remove os registros com valores NA e remove as duas primeiras colunas (não são necessárias)
dataset = dataset.iloc[0:, 2:9].dropna()
print("Dados:\n", dataset.head())

# Checando se há valores missing
missing_values = dataset.isnull().values.any()
print("Valores missing: ", missing_values)

# Obtém os valores dos atributos
dataset_atrib = dataset.values
print("Atributos:\n", dataset_atrib)

# Coleta uma amostra de 1% dos dados para não comprometer a memória do computador
amostra1, amostra2 = train_test_split(dataset_atrib, train_size=0.01)
print("Shape da amostra: ", amostra1.shape)

# Aplica redução de dimensionalidade
pca = PCA(n_components=2).fit_transform(amostra1)

# Determinando um range de K
k_range = range(1, 12)

# Aplicando o modelo K-Means para cada valor de K (pode levar bastante tempo para ser executada)
k_means_var = [KMeans(n_clusters=k).fit(pca) for k in k_range]

# Ajustando o centróide do cluster para cada modelo
centroids = [X.cluster_centers_ for X in k_means_var]

# Calculando a distância euclidiana de cada ponto de dado para o centróide
k_euclid = [cdist(pca, cent, "euclidean") for cent in centroids]
dist = [np.min(ke, axis=1) for ke in k_euclid]

# Soma dos quadrados das distâncias dentro do cluster
soma_quadrados_intra_cluster = [sum(d**2) for d in dist]

# Soma total dos quadrados
soma_total = sum(pdist(pca) ** 2) / pca.shape[0]

# Soma dos quadrados entre clusters
soma_quadrados_inter_cluster = soma_total - soma_quadrados_intra_cluster

# Curva de Elbow
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(k_range, soma_quadrados_inter_cluster / soma_total * 100, "b*-")
ax.set_ylim((0, 100))
plt.grid(True)
plt.xlabel("Número de Clusters")
plt.ylabel("Percentual de Variância Explicada")
plt.title("Variância Explicada x Valor de K")
plt.savefig("curva-de-elbow.png")
plt.close()


# Criando um modelo com K = 8
modelo_v1 = KMeans(n_clusters=8)
modelo_v1.fit(pca)

# Obtém os valores mínimos e máximos e organiza o shape
x_min, x_max = pca[:, 0].min() - 5, pca[:, 0].max() - 1
y_min, y_max = pca[:, 1].min() + 1, pca[:, 1].max() + 5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = modelo_v1.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot das áreas dos clusters
plt.figure(1)
plt.clf()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Paired,
    aspect="auto",
    origin="lower",
)
plt.savefig("area-clusters-v1.png")
plt.close()

# Plot dos centróides
plt.plot(pca[:, 0], pca[:, 1], "k.", markersize=4)
centroids = modelo_v1.cluster_centers_
inert = modelo_v1.inertia_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=169,
    linewidths=3,
    color="r",
    zorder=8,
)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.savefig("centroides-v1.png")
plt.close()

# Silhouette Score
labels = modelo_v1.labels_
s_score = silhouette_score(pca, labels, metric="euclidean")
print("Silhouette Score v1: ", s_score)


# Criando um modelo com K = 10
modelo_v2 = KMeans(n_clusters=10)
modelo_v2.fit(pca)

# Obtém os valores mínimos e máximos e organiza o shape
x_min, x_max = pca[:, 0].min() - 5, pca[:, 0].max() - 1
y_min, y_max = pca[:, 1].min() + 1, pca[:, 1].max() + 5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = modelo_v2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot das áreas dos clusters
plt.figure(1)
plt.clf()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Paired,
    aspect="auto",
    origin="lower",
)
plt.savefig("area-clusters-v2.png")
plt.close()

# Plot dos centróides
plt.plot(pca[:, 0], pca[:, 1], "k.", markersize=4)
centroids = modelo_v2.cluster_centers_
inert = modelo_v2.inertia_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=169,
    linewidths=3,
    color="r",
    zorder=8,
)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.savefig("centroides-v2.png")
plt.close()

# Silhouette Score
labels = modelo_v2.labels_
s_score = silhouette_score(pca, labels, metric="euclidean")
print("Silhouette Score v1: ", s_score)


# Criando o Cluster Map com os clusters do Modelo V1 que apresentou melhor Silhouette Score.
# Lista com nomes das colunas
names = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

# Cria o cluster map
cluster_map = pd.DataFrame(amostra1, columns=names)
cluster_map["Global_active_power"] = pd.to_numeric(cluster_map["Global_active_power"])
cluster_map["cluster"] = modelo_v1.labels_
print("Cluster Map:\n", cluster_map)


# Calcula a média de consumo de energia por cluster
mean_energy = cluster_map.groupby("cluster")["Global_active_power"].mean()
print("Média de consumo de energia por cluster:\n", mean_energy)
