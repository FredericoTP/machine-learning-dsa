# Lab - Collocations e Processamento de Comentários de Avaliações de Hotéis

# Definição do Problema
# Dado um conjunto de texto de avaliações (comentários) de hotéis, vamos buscar
# as Collocations mais relevantes que ajudam a explicar as avaliações!

# Imports
import pandas as pd
import nltk
import spacy
import re
import string
from nltk.corpus import stopwords

# Se necessário, faça o download das stopwords
# nltk.download('stopwords')

# Carregando dados de avaliações de hotéis
# Fonte de dados: https://datafiniti.co/products/business-data/
avaliacoes_hoteis = pd.read_csv(
    "https://raw.githubusercontent.com/dsacademybr/Datasets/master/dataset7.csv"
)

# Visualiza os dados
print("Dados:\n", avaliacoes_hoteis.head(5))

# Tipo do objeto
print("Tipo do objeto: ", type(avaliacoes_hoteis))

# Shape
print("Shape: ", avaliacoes_hoteis.shape)

# Extraindo as avaliações
comentarios = avaliacoes_hoteis["reviews.text"]

# Converte para o tipo string
comentarios = comentarios.astype("str")


# Função para remover caracteres non-ascii
def removeNoAscii(s):
    return "".join(i for i in s if ord(i) < 128)


# Remove caracteres non-ascii
comentarios = comentarios.map(lambda x: removeNoAscii(x))

# Obtém as stopwords em todos os idiomas
dicionario_stopwords = {
    lang: set(nltk.corpus.stopwords.words(lang))
    for lang in nltk.corpus.stopwords.fileids()
}


# Função para detectar o idioma predominante com base nas stopwords
def descobre_idioma(text):

    # Aplica tokenização considerando pontuação
    palavras = set(nltk.wordpunct_tokenize(text.lower()))

    # Conta o total de palavras tokenizadas considerando o dicionário de stopwords
    lang = max(
        (
            (lang, len(palavras & stopwords))
            for lang, stopwords in dicionario_stopwords.items()
        ),
        key=lambda x: x[1],
    )[0]

    # Verifica se o idioma é português
    if lang == "portuguese":
        return True
    else:
        return False


# Filtra somente os comentários em português
comentarios_portugues = comentarios[comentarios.apply(descobre_idioma)]

# Shape
print("Shape comentários em PT: ", comentarios_portugues.shape)

# Print
print("Comentários em PT:\n", comentarios_portugues)


# Função para detectar o idioma predominante com base nas stopwords
def descobre_idioma(text):
    words = set(nltk.wordpunct_tokenize(text.lower()))
    lang = max(
        (
            (lang, len(words & stopwords))
            for lang, stopwords in dicionario_stopwords.items()
        ),
        key=lambda x: x[1],
    )[0]
    if lang == "english":
        return True
    else:
        return False


# Filtra somente os comentários em inglês
comentarios_ingles = comentarios[comentarios.apply(descobre_idioma)]

# Shape
print("Shape comentários em EN: ", comentarios_ingles.shape)

# Print
print("Comentários em EN:\n", comentarios_ingles.head())

# Removendo duplicidades
comentarios_ingles.drop_duplicates(inplace=True)

# Shape
print("Shape comentários em EN: ", comentarios_ingles.shape)

# Baixando o dicionário inglês
# https://spacy.io/usage/models
# !python -m spacy download en_core_web_sm

# Carrega o dcionário em nossa sessão SpaCy
nlp = spacy.load("en_core_web_sm")


# Função para limpar e lematizar os comentários
def limpa_comentarios(text):

    # Remove pontuação usando expressão regular
    regex = re.compile("[" + re.escape(string.punctuation) + "\\r\\t\\n]")
    nopunct = regex.sub(" ", str(text))

    # Usa o SpaCy para lematização
    doc = nlp(nopunct, disable=["parser", "ner"])
    lemma = [token.lemma_ for token in doc]
    return lemma


# Aplica a função aos dados
comentarios_ingles_lemmatized = comentarios_ingles.map(limpa_comentarios)

# Coloca tudo em minúsculo
comentarios_ingles_lemmatized = comentarios_ingles_lemmatized.map(
    lambda x: [word.lower() for word in x]
)

# Visualiza os dados
print(comentarios_ingles_lemmatized.head())

# Vamos tokenizar os comentários
comentarios_tokens = [item for items in comentarios_ingles_lemmatized for item in items]

# Tokens
print(comentarios_tokens)


# ------------------------------------------------------


# Estratégia 1 - Buscando Bigramas e Trigramas Mais Relevantes nos Comentários Por Frequência

# Métricas de associação de bigramas (esse objeto possui diversos atributos, como freq, pmi, teste t, etc...)
bigramas = nltk.collocations.BigramAssocMeasures()

# Métricas de associação de trigramas
trigramas = nltk.collocations.TrigramAssocMeasures()

# O próximo passo é criar um buscador de bigramas nos tokens
buscaBigramas = nltk.collocations.BigramCollocationFinder.from_words(comentarios_tokens)

# Fazemos o mesmo com trigramas. Fique atento aos métodos que estão sendo usados
buscaTrigramas = nltk.collocations.TrigramCollocationFinder.from_words(
    comentarios_tokens
)

# Vamos contar quantas vezes cada bigrama aparece nos tokens dos comentários
bigrama_freq = buscaBigramas.ngram_fd.items()

# Frequência dos bigramas
bigrama_freq

# Vamos converter o dicionário anterior em uma tabela de frequência no formato do Pandas para os bigramas
FreqTabBigramas = pd.DataFrame(
    list(bigrama_freq), columns=["Bigrama", "Freq"]
).sort_values(by="Freq", ascending=False)

# Visualiza a tabela
print(FreqTabBigramas.head(5))

# Vamos contar quantas vezes cada trigrama aparece nos tokens dos comentários
trigrama_freq = buscaTrigramas.ngram_fd.items()

# Tabela de frequência no formato do Pandas para os trigramas
FreqTabTrigramas = pd.DataFrame(
    list(trigrama_freq), columns=["Trigrama", "Freq"]
).sort_values(by="Freq", ascending=False)

# Visualiza a tabela
print(FreqTabTrigramas.head(5))

# Vamos criar uma lista de stopwords
en_stopwords = set(stopwords.words("english"))


# Função para filtrar bigramas ADJ/NN e remover stopwords
def filtra_tipo_token_bigrama(ngram):

    # Verifica se é pronome
    if "-pron-" in ngram or "t" in ngram:
        return False

    # Loop nos ngramas para verificar se é stopword
    for word in ngram:
        if word in en_stopwords or word.isspace():
            return False

    # Tipos de tokens aceitáveis
    acceptable_types = ("JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS")

    # Subtipos
    second_type = ("NN", "NNS", "NNP", "NNPS")

    # Tags
    tags = nltk.pos_tag(ngram)

    # Retorna o que queremos, ADJ/NN
    if tags[0][1] in acceptable_types and tags[1][1] in second_type:
        return True
    else:
        return False


# Agora filtramos os bigramas
bigramas_filtrados = FreqTabBigramas[
    FreqTabBigramas.Bigrama.map(lambda x: filtra_tipo_token_bigrama(x))
]

# Visualiza a tabela
print(bigramas_filtrados.head(5))


# Função para filtrar trigramas ADJ/NN e remover stopwords
def filtra_tipo_token_trigrama(ngram):

    # Verifica se é pronome
    if "-pron-" in ngram or "t" in ngram:
        return False

    # Loop nos ngramas para verificar se é stopword
    for word in ngram:
        if word in en_stopwords or word.isspace():
            return False

    # Tipos de tokens aceitáveis
    first_type = ("JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS")

    # Subtipos
    second_type = ("JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS")

    # Tags
    tags = nltk.pos_tag(ngram)

    # Retorna o que queremos, ADJ/NN
    if tags[0][1] in first_type and tags[2][1] in second_type:
        return True
    else:
        return False


# Agora filtramos os trigramas
trigramas_filtrados = FreqTabTrigramas[
    FreqTabTrigramas.Trigrama.map(lambda x: filtra_tipo_token_trigrama(x))
]

# Visualiza a tabela
print(trigramas_filtrados.head(5))


# ------------------------------------------------------


# Estratégia 2 - Buscando Bigramas e Trigramas Mais Relevantes nos Comentários Por PMI

# Vamos retornar somente bigramas com 20 ou mais ocorrências
print(buscaBigramas.apply_freq_filter(20))

# Criamos a tabela
PMITabBigramas = pd.DataFrame(
    list(buscaBigramas.score_ngrams(bigramas.pmi)), columns=["Bigrama", "PMI"]
).sort_values(by="PMI", ascending=False)

# Visualiza a tabela
print(PMITabBigramas.head(5))

# Vamos retornar somente trigramas com 20 ou mais ocorrências
print(buscaTrigramas.apply_freq_filter(20))

# Criamos a tabela
PMITabTrigramas = pd.DataFrame(
    list(buscaTrigramas.score_ngrams(trigramas.pmi)), columns=["Trigrama", "PMI"]
).sort_values(by="PMI", ascending=False)

# Visualiza a tabela
print(PMITabTrigramas.head(5))


# ------------------------------------------------------


# Estratégia 3 - Buscando Bigramas e Trigramas Mais Relevantes nos Comentários Por Teste t

# Criamos a tabela para os bigramas
# Observe como o resultado do teste t é obtido: buscaBigramas.score_ngrams(bigramas.student_t)
TestetTabBigramas = pd.DataFrame(
    list(buscaBigramas.score_ngrams(bigramas.student_t)), columns=["Bigrama", "Teste-t"]
).sort_values(by="Teste-t", ascending=False)

# Vamos aplicar o filtro pelo tipo de token conforme aplicamos no método 1
bigramas_t_filtrados = TestetTabBigramas[
    TestetTabBigramas.Bigrama.map(lambda x: filtra_tipo_token_bigrama(x))
]

# Visualiza a tabela
print(bigramas_t_filtrados.head(5))

# Criamos a tabela para os trigramas
TestetTabTrigramas = pd.DataFrame(
    list(buscaTrigramas.score_ngrams(trigramas.student_t)),
    columns=["Trigrama", "Teste-t"],
).sort_values(by="Teste-t", ascending=False)

# Vamos aplicar o filtro pelo tipo de token conforme aplicamos no método 1
trigramas_t_filtrados = TestetTabTrigramas[
    TestetTabTrigramas.Trigrama.map(lambda x: filtra_tipo_token_trigrama(x))
]

# Visualiza a tabela
print(trigramas_t_filtrados.head(5))


# ------------------------------------------------------


# Estratégia 4 - Buscando Bigramas e Trigramas Mais Relevantes nos Comentários Por Teste do Qui-quadrado

# Prepara a tabela
# Observe como estamos coletando a estatística qui-quadrado: buscaBigramas.score_ngrams(bigramas.chi_sq)
QuiTabBigramas = pd.DataFrame(
    list(buscaBigramas.score_ngrams(bigramas.chi_sq)), columns=["Bigrama", "Qui"]
).sort_values(by="Qui", ascending=False)

# Visualiza a tabela
print(QuiTabBigramas.head(5))

# Prepara a tabela
QuiTabTrigramas = pd.DataFrame(
    list(buscaTrigramas.score_ngrams(trigramas.chi_sq)), columns=["Trigrama", "Qui"]
).sort_values(by="Qui", ascending=False)

# Visualiza a tabela
print(QuiTabTrigramas.head(5))


# ------------------------------------------------------


# Comparação e Resultado Final

# Vamos extrair os 10 Collocations bigramas mais relevantes de acordo com cada um dos 4 métodos usados
# Lembre-se que aplicamos filtros para remover as stopwords e devemos usar a tabela filtrada
metodo1_bigrama = bigramas_filtrados[:10].Bigrama.values
metodo2_bigrama = PMITabBigramas[:10].Bigrama.values
metodo3_bigrama = bigramas_t_filtrados[:10].Bigrama.values
metodo4_bigrama = QuiTabBigramas[:10].Bigrama.values

# Vamos extrair os 10 Collocations trigramas mais relevantes de acordo com cada um dos 4 métodos usados
# Lembre-se que aplicamos filtros para remover as stopwords e devemos usar a tabela filtrada
metodo1_trigrama = trigramas_filtrados[:10].Trigrama.values
metodo2_trigrama = PMITabTrigramas[:10].Trigrama.values
metodo3_trigrama = trigramas_t_filtrados[:10].Trigrama.values
metodo4_trigrama = QuiTabTrigramas[:10].Trigrama.values

# Vamos criar um super dataframe com todos os resultados para bigramas
comparaBigramas = pd.DataFrame(
    [metodo1_bigrama, metodo2_bigrama, metodo3_bigrama, metodo4_bigrama]
).T

# Nossa tabela precisa de nomes para as colunas
comparaBigramas.columns = ["Frequência", "PMI", "Teste-t", "Teste Qui-quadrado"]

# Visualiza a tabela - Padrão CSS
print(
    comparaBigramas.style.set_properties(
        **{"background-color": "green", "color": "white", "border-color": "white"}
    )
)

# Vamos criar um super dataframe com todos os resultados para trigramas
comparaTrigramas = pd.DataFrame(
    [metodo1_trigrama, metodo2_trigrama, metodo3_trigrama, metodo4_trigrama]
).T

# Nossa tabela precisa de nomes para as colunas
comparaTrigramas.columns = ["Frequência", "PMI", "Teste-t", "Teste Qui-quadrado"]

# Visualiza a tabela
print(
    comparaTrigramas.style.set_properties(
        **{"background-color": "blue", "color": "white", "border-color": "white"}
    )
)


# ------------------------------------------------------


# Conclusão

# Podemos ver que os métodos PMI e Qui-quadrado fornecem bons resultados. Seus resultados também são semelhantes.

# Mas os métodos de Frequência e Teste-t apresentam os melhores resutados e são também muito semelhantes entre si.
