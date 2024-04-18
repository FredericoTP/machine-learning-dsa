import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Criando uma Decision Tree

# Carrega o dataset
irisData = pd.read_csv("iris_data.csv")

# Visualiza as primeiras linhas
print(irisData.head())

# Resumo estatístico
print(irisData.describe())

# Correlação
# print(irisData.corr())

# Atributos e Variável target
features = irisData[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]
targetVariables = irisData.Class

# Gera os dados de treino
featureTrain, featureTest, targetTrain, targetTest = train_test_split(
    features, targetVariables, test_size=0.2
)

# Criação do modelo
clf = DecisionTreeClassifier()

modelo = clf.fit(featureTrain, targetTrain)
previsoes = modelo.predict(featureTest)

print("Confusion Matrix: ", confusion_matrix(targetTest, previsoes))
print("Accuracy Score: ", accuracy_score(targetTest, previsoes))
