# Imports
import os
import re
import nltk
import spacy
import string
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, LancasterStemmer, RegexpStemmer, WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus import gutenberg as gt

# Instalando os arquivos de dados e dicionários do NLTK
# nltk.download("all")


# ---------------------------------------------------------


# Tokenization

# Dividindo um Parágrafo em Frases

paragrafo = "Seja Bem-vindo a Data Science Academy. Bom saber que você está aprendendo PLN. Obrigado por estar conosco."
tokens = sent_tokenize(paragrafo)
print("Parágrafo tokenizada:\n", tokens)

# Utilizando dicionário do pacote NLTK
tokenizer = nltk.data.load("tokenizers/punkt/portuguese.pickle")
tokens = tokenizer.tokenize(paragrafo)
print("Parágrafo tokenizada:\n", tokens)

# Dicionário em espanhol
spanish_tokenizer = nltk.data.load("tokenizers/punkt/spanish.pickle")
tokens = spanish_tokenizer.tokenize("Hola amigo. Estoy bien.")
print("Parágrafo tokenizada:\n", tokens)


# Dividindo uma Frase em Palavras
tokens = word_tokenize("Data Science Academy")
print("Palavras tokenizada:\n", tokens)

tokenizer = TreebankWordTokenizer()
print("Palavras tokenizada:\n", tokenizer.tokenize("Inteligência Artificial"))

print("Palavras tokenizada:\n", word_tokenize("can't"))

tokenizer = WordPunctTokenizer()
print("Palavras tokenizada:\n", tokenizer.tokenize("Can't is a contraction."))

tokenizer = RegexpTokenizer("[\w']+")
print("Palavras tokenizada:\n", tokenizer.tokenize("Can't is a contraction."))

tokens = regexp_tokenize("Can't is a contraction.", "[\w']+")
print("Palavras tokenizada:\n", tokens)

tokenizer = RegexpTokenizer("\s+", gaps=True)
print("Palavras tokenizada:\n", tokenizer.tokenize("Can't is a contraction."))

# Uma operação única com List Comprehension

# Texto a ser tokenizado
texto = "Seja Bem-vindo a Data Science Academy. Bom saber que você está aprendendo PLN. Obrigado por estar conosco."

# List Comprehension
print([word_tokenize(frase) for frase in sent_tokenize(texto)])


# ---------------------------------------------------------


# Stopwords

english_stops = set(stopwords.words("english"))
palavras = ["Can't", "is", "a", "contraction"]
print(
    "Palavras sem stopwords:\n",
    [palavra for palavra in palavras if palavra not in english_stops],
)

portuguese_stops = set(stopwords.words("portuguese"))
palavras = ["Aquilo", "é", "uma", "ferrari"]
print(
    "Palavras sem stopwords:\n",
    [palavra for palavra in palavras if palavra not in portuguese_stops],
)


# ---------------------------------------------------------


# Stemming

porter_stemmer = PorterStemmer()
print("Stemming: ", porter_stemmer.stem("cooking"))
print("Stemming: ", porter_stemmer.stem("cookery"))

lanc_stemmer = LancasterStemmer()
print("Stemming: ", lanc_stemmer.stem("cooking"))
print("Stemming: ", lanc_stemmer.stem("cookery"))

regexp_stemmer = RegexpStemmer("ing")
print("Stemming: ", regexp_stemmer.stem("cooking"))

lista_palavras = [
    "cat",
    "cats",
    "know",
    "knowing",
    "time",
    "timing",
    "football",
    "footballers",
]
porter_stemmer = PorterStemmer()
for palavra in lista_palavras:
    print(palavra + " -> " + porter_stemmer.stem(palavra))


def SentenceStemmer(sentence):
    tokens = word_tokenize(sentence)
    stems = [porter_stemmer.stem(token) for token in tokens]
    return " ".join(stems)


print(SentenceStemmer("The cats and dogs are running"))


# ---------------------------------------------------------


# Lemmatization

# nltk.download("wordnet")

wordnet_lemmatizer = WordNetLemmatizer()
print(wordnet_lemmatizer.lemmatize("mice"))
print(
    wordnet_lemmatizer.lemmatize("cacti")
)  # plural da palavra cactus - cactuses (inglês) ou cacti (latin)
print(wordnet_lemmatizer.lemmatize("horses"))
print(wordnet_lemmatizer.lemmatize("wolves"))

print(wordnet_lemmatizer.lemmatize("madeupwords"))
print(porter_stemmer.stem("madeupwords"))


def return_word_pos_tuples(sentence):
    return pos_tag(word_tokenize(sentence))


sentence = "The cats and dogs are running"
print(return_word_pos_tuples(sentence))


def get_pos_wordnet(pos_tag):
    pos_dict = {
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "J": wordnet.ADJ,
        "R": wordnet.ADV,
    }

    return pos_dict.get(pos_tag[0].upper(), wordnet.NOUN)


print(get_pos_wordnet("VBG"))


def lemmatize_with_pos(sentence):
    new_sentence = []
    tuples = return_word_pos_tuples(sentence)
    for tup in tuples:
        pos = get_pos_wordnet(tup[1])
        lemma = wordnet_lemmatizer.lemmatize(tup[0], pos=pos)
        new_sentence.append(lemma)
    return new_sentence


print(lemmatize_with_pos(sentence))


# ---------------------------------------------------------


# Corpus

corpusdir = "corpus/"

newcorpus = PlaintextCorpusReader(corpusdir, ".*")

newcorpus.fileids()

# Acessando cada arquivo no Corpus com loop for
for infile in sorted(newcorpus.fileids()):
    print(infile)

# Acessando cada arquivo no Corpus com list comprehension
[arquivo for arquivo in sorted(newcorpus.fileids())]

# Conteúdo do Corpus
print(newcorpus.raw().strip())

# Conteúdo do Corpus tokenizado por parágrafo (nova linha)
print(newcorpus.paras())

# Conteúdo do Corpus tokenizado por sentença (nova linha)
print(newcorpus.sents())

# Conteúdo do Corpus tokenizado por sentença por arquivo
print(newcorpus.sents(newcorpus.fileids()[0]))

print(gt.fileids())

shakespeare_macbeth = gt.words("shakespeare-macbeth.txt")
print(shakespeare_macbeth)

raw = gt.raw("shakespeare-macbeth.txt")
print(raw)

sents = gt.sents("shakespeare-macbeth.txt")
print(sents)

for fileid in gt.fileids():
    num_words = len(gt.words(fileid))
    num_sents = len(gt.sents(fileid))
    print("Dados do Arquivo:", fileid)
    print("Número de Palavras:", num_words)
    print("Número de Frases:", num_sents, end="\n\n\n")
