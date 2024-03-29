# Pacotes
install.packages("tm")
install.packages("SnowballC")
install.packages("wordcloud")
install.packages("gmodels")
library(tm)
library(SnowballC)
library(wordcloud)
library(e1071)
library(gmodels)

# Carregando os dados
dados <- read.csv("sms_spam.csv", stringsAsFactors = FALSE)

# Examinando a estrutura dos dados
str(dados)

# Convertendo para fator
dados$type <- factor(dados$type)

# Examinando a estrutura dos dados
str(dados$type)
table(dados$type)

# Contruindo um Corpus
dados_corpus <- VCorpus(VectorSource(dados$text))

# Examinando a estrutura de dados
print(dados_corpus)
inspect(dados_corpus[1:2])

# Ajustando a estrutura
as.character(dados_corpus[[1]])
lapply(dados_corpus[1:2], as.character)

# Limpeza do Corpus com tm_map()
?tm_map
dados_corpus_clean <- tm_map(dados_corpus, content_transformer(tolower))

# Diferenças entre o Corpus inicial e o Corpus após a limpeza
as.character(dados_corpus[[1]])
as,character(dados_corpus_clean[[1]])

# Outras etapas de limpeza
dados_corpus_clean <- tm_map(dados_corpus_clean, removeNumbers) # remove números
dados_corpus_clean <- tm_map(dados_corpus_clean, removeWords, stopwords()) # remove stop words
dados_corpus_clean <- tm_map(dados_corpus_clean, removePunctuation) # remove pontuação

# Criando uma função para substituir ao invés de remover pontuação
removePunctuation("hello...world")
replacePunctuation <- function(x) { gsub("[[:punct:]]+", " ", x) }
replacePunctuation("hello...world")

# word stemming
?wordStem
wordStem(c("learn", "learned", "learning", "learns"))

# Aplicando Stem
dados_corpus_clean <- tm_map(dados_corpus_clean, stemDocument)

# Eliminando espaço em branco
dados_corpus_clean <- tm_map(dados_corpus_clean, stripWhitespace)

# Examinando a versão final do Corpus
lapply(dados_corpus[1:3], as.character)
lapply(dados_corpus_clean[1:3], as.character)

# Criando uma matriz esparsa document-term
?DocumentTermMatrix
dados_dtm <- DocumentTermMatrix(dados_corpus_clean)

# Solução alternativa 2 - cria uma matriz esparsa document-term direto a partis do Corpus
dados_dtm2 <- DocumentTermMatrix(dados_corpus, control = list(tolower = TRUE, removeNumbers = TRUE, stopwords = TRUE, removePunctuation = TRUE, stemming = TRUE))

# Solução alternativa 3 - usando stop words customizadas a partir da função
dados_dtm3 <- DocumentTermMatrix(dados_corpus, control = list(tolower = TRUE, removeNumbers = TRUE, stopwords = function(x) { removeWords(x, stopwords()) }, removePunctuation = TRUE, stemming = TRUE))

# Comparando os resultados
dados_dtm
dados_dtm2
dados_dtm3

# Criando datasets de treino e de teste
dados_dtm_train <- dados_dtm[1:4169, ]
dados_dtm_test <- dados_dtm[4170:5559, ]

# Labels (variável target)
dados_train_labels <- dados[1:4169, ]$type
dados_test_labels <- dados[4170:5559, ]$type

# Verificando se a proporção de Spam é similar
prop.table(table(dados_train_labels))
prop.table(table(dados_test_labels))

# Word Cloud
wordcloud(dados_corpus_clean, min.freq = 50, random.order = FALSE)

# Frequência dos dados
sms_dtm_freq_train <- removeSparseTerms(dados_dtm_train, 0.999)
sms_dtm_freq_train

# Indicador de Features para palavras frequentes
findFreqTerms(dados_dtm_train, 5)

# save frequently-appearing terms to a character vector
sms_freq_words <- findFreqTerms(dados_dtm_train, 5)
str(sms_freq_words)

# Criando subsets apenas com palavras mais frequentes
sms_dtm_freq_train <- dados_dtm_train[ , sms_freq_words]
sms_dtm_freq_test <- dados_dtm_test[ , sms_freq_words]

# Converte para fator
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

# apply() converte counts para colunas de daos de treino e de teste
sms_train <- apply(sms_dtm_freq_train, MARGIN = 2, convert_counts)
sms_test <- apply(sms_dtm_freq_test, MARGIn = 2, convert_counts)

# Treinando o modelo
?naiveBayes
nb_classifier <- naiveBayes(sms_train, dados_train_labels)

# Avaliando o modelo
sms_test_pred <- predict(nb_classifier, sms_test)

# Confusion Matrix
CrossTable(sms_test_pred, dados_test_labels, prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE, dnn = c("Previsto", "Observado"))

# Melhorando a performance do modelo aplicando suavização laplace
nb_classifier_v2 <- naiveBayes(sms_train, dados_train_labels, laplace = 1)

sms_test_pred2 <- predict(nb_classifier_v2, sms_test)

CrossTable(sms_test_pred2, dados_test_labels, prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE, dnn = c("Previsto", "Observado"))