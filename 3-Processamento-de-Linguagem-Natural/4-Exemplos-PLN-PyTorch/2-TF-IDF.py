# TF-IDF Para Identificação das Palavras Mais Relevantes em Um Livro

# Imports
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize


# Preparando os Dados

# Carrega os dados
dados_livro_emma = nltk.corpus.gutenberg.sents("austen-emma.txt")

# Listas para receber as frases e as palavras do texto
dados_livro_emma_frases = []
dados_livro_emma_palavras = []

# Loop para a tokenização
# https://docs.python.org/3/library/stdtypes.html
for sentence in dados_livro_emma:
    dados_livro_emma_frases.append(
        [word.lower() for word in sentence if word.isalpha()]
    )
    for word in sentence:
        if word.isalpha():
            dados_livro_emma_palavras.append(word.lower())

# Vamos converter a lista de palavras em um conjunto (set)
dados_livro_emma_palavras = set(dados_livro_emma_palavras)

# Visualiza as frases
print(dados_livro_emma_frases)

# Visualiza as palavras
print(dados_livro_emma_palavras)


# Frequência do termo


# Função para calcular a Termo Frequência
def TermFreq(documento, palavra):
    doc_length = len(documento)
    ocorrencias = len([w for w in documento if w == palavra])
    return ocorrencias / doc_length


# Aplica a função
TermFreq(dados_livro_emma_frases[5], "mother")


# Criamos um corpus Bag of words
def cria_dict():
    output = {}
    for word in dados_livro_emma_palavras:
        output[word] = 0
        for doc in dados_livro_emma_frases:
            if word in doc:
                output[word] += 1
    return output


# Cria o dicionário
df_dict = cria_dict()

# Filtra o dicionário
print(df_dict["mother"])


# Frequência Inversa


# Função para calcular a Frequência Inversa de Documentos
def InverseDocumentFrequency(word):
    N = len(dados_livro_emma_frases)
    try:
        df = df_dict[word] + 1
    except:
        df = 1
    return np.log(N / df)


# Aplica a função
print(InverseDocumentFrequency("mother"))


# TF/IDF


# Função TF-IDF
def TFIDF(doc, word):
    tf = TermFreq(doc, word)
    idf = InverseDocumentFrequency(word)
    return tf * idf


# Print
print("mother: " + str(TFIDF(dados_livro_emma_frases[5], "mother")))

# Print
print("mother: " + str(TFIDF(dados_livro_emma_frases[30], "mother")))
