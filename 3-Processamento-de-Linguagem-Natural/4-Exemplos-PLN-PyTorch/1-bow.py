# Modelo de Classificação de Idiomas de Sentenças com Bag of Words e PyTorch

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn, optim

# Dados de treino
dados_treino = [
    ("Tenho vinte paginas de leitura".lower().split(), "Portuguese"),
    ("I will visit the library".lower().split(), "English"),
    ("I am reading a book".lower().split(), "English"),
    ("This is my favourite chapter".lower().split(), "English"),
    ("Estou na biblioteca lendo meu livro preferido".lower().split(), "Portuguese"),
    ("Gosto de livros sobre viagens".lower().split(), "Portuguese"),
]
print("Dados de treino:\n", dados_treino)

# Dados de teste
dados_teste = [
    ("Estou lendo".lower().split(), "Portuguese"),
    ("This is not my favourite book".lower().split(), "English"),
]
print("Dados de teste:\n", dados_teste)

# Prepara o dicionário do vocabulário

# Dicionário para o vocabulário
dict_vocab = {}

# Contadoor
i = 0

# Loop pelos dados de treino e teste
for palavras, idiomas in dados_treino + dados_teste:
    for palavra in palavras:
        if palavra not in dict_vocab:
            dict_vocab[palavra] = i
            i += 1

# Visualiza o vocabulário
print(dict_vocab)

# Tamanho do corpus
tamanho_corpus = len(dict_vocab)
print("Tamanho do corpus: ", tamanho_corpus)

# Número de idiomas
idiomas = 2

# Índice para os idiomas
label_index = {"Portuguese": 0, "English": 1}


# ------------------------------------------------


# Construção do Modelo


# Classe para o modelo BOW de classificação
class ModeloBOW(nn.Module):

    # Método construtor
    def __init__(self, lista_idiomas, tamanho_do_corpus):
        super(ModeloBOW, self).__init__()
        self.linear = nn.Linear(tamanho_do_corpus, lista_idiomas)

    # Feed Forward
    def forward(self, bow_vec):
        return F.log_softmax(self.linear(bow_vec), dim=1)


# Função para criar o vetor BOW necessário para o treinamento
def cria_bow_vetor(sentence, word_index):
    word_vec = torch.zeros(tamanho_corpus)
    for word in sentence:
        word_vec[dict_vocab[word]] += 1
    return word_vec.view(1, -1)


# Função para criar a variável target
def cria_target(label, label_index):
    return torch.LongTensor([label_index[label]])


# Cria o modelo
modelo = ModeloBOW(idiomas, tamanho_corpus)

# Função de perda (loss)
loss_function = nn.NLLLoss()

# Otimizador
optimizer = optim.SGD(modelo.parameters(), lr=0.1)


# ------------------------------------------------


# Treinamento do Modelo

# Loop de treinamentoo
for epoch in range(100):

    for sentence, label in dados_treino:

        modelo.zero_grad()

        bow_vec = cria_bow_vetor(sentence, dict_vocab)
        target = cria_target(label, label_index)

        log_probs = modelo(bow_vec)

        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print("Epoch: ", str(epoch + 1), ", Loss: " + str(loss.item()))


# ------------------------------------------------


# Previsões e Avaliação do Modelo


# Função para previsões
def faz_previsao(data):

    with torch.no_grad():
        sentence = data[0]
        label = data[1]
        bow_vec = cria_bow_vetor(sentence, dict_vocab)
        log_probs = modelo(bow_vec)
        print(sentence)
        print(
            "Probabilidade de ser o label: " + label, "é igual a: ", np.exp(log_probs)
        )


# Previsão com a primeira sentença de teste
print(faz_previsao(dados_teste[0]))

# Previsão com a segunda sentença de teste
print(faz_previsao(dados_teste[1]))


# Previsões com Novas Frases

# Nova frase
novas_frases = [
    ("Tenho livros sobre viagens".lower().split(), "Portuguese"),
    ("Estou escrevendo".lower().split(), "Portuguese"),
    ("Gosto de biblioteca".lower().split(), "Portuguese"),
]

print(faz_previsao(novas_frases[0]))
print(faz_previsao(novas_frases[1]))  # Error por palavra não estar no dicionário
print(faz_previsao(novas_frases[2]))
