import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydot
import graphviz
from pathlib import Path

# Criando o dataset
instancias = [
    {"Melhor Amigo": False, "Especie": "Cachorro"},
    {"Melhor Amigo": True, "Especie": "Cachorro"},
    {"Melhor Amigo": True, "Especie": "Gato"},
    {"Melhor Amigo": True, "Especie": "Gato"},
    {"Melhor Amigo": False, "Especie": "Gato"},
    {"Melhor Amigo": True, "Especie": "Gato"},
    {"Melhor Amigo": True, "Especie": "Gato"},
    {"Melhor Amigo": False, "Especie": "Cachorro"},
    {"Melhor Amigo": True, "Especie": "Gato"},
    {"Melhor Amigo": False, "Especie": "Cachorro"},
    {"Melhor Amigo": False, "Especie": "Cachorro"},
    {"Melhor Amigo": False, "Especie": "Gato"},
    {"Melhor Amigo": True, "Especie": "Gato"},
    {"Melhor Amigo": True, "Especie": "Cachorro"},
]

# Transformando o Ddicionário em DataFrame
df = pd.DataFrame(instancias)
print(df)

# Preparando os dados de treino e de teste
X_train = [[1] if a else [0] for a in df["Melhor Amigo"]]
y_train = [1 if d == "Cachorro" else 0 for d in df["Especie"]]
labels = ["Melhor Amigo"]
print(X_train)
print(y_train)

# Construindo o Classificador Baseado em Entropia
modelo_v1 = DecisionTreeClassifier(
    max_depth=None,
    max_features=None,
    criterion="entropy",
    min_samples_leaf=1,
    min_samples_split=2,
)

# Apresentando os dados ao Classificador
modelo_v1.fit(X_train, y_train)

# Definindo o nome do arquivo com a árvore de decisão
diretorio_atual = Path().resolve()
arquivo = str(diretorio_atual) + "/tree_modelo_v1.dot"
print(arquivo)

# Gerando o gráfico da árvore de decisão
export_graphviz(modelo_v1, out_file=arquivo, feature_names=labels)
with open(arquivo) as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

# !dot -Tpng tree_modelo_v1.dot -o tree_modelo_v1.png
