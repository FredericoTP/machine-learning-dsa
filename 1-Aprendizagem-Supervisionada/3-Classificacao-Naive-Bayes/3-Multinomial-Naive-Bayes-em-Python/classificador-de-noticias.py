# O classificador Multinomial Naive Bayes é adequado para classificação
# com variáveis discretas (por exemplo, contagens de palavras para a
# classificação de texto). A distribuição multinomial normalmente requer
# contagens de entidades inteiras. No entanto, na prática, contagens
# fracionadas como tf-idf também podem funcinar.

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

# Definindo as categorias
# (usando apenas 4 de um total de 20 disponíveis para que o
# proceso de classificação seja mais rápido)
categories = ["alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med"]

# Treinamento
twenty_train = fetch_20newsgroups(
    subset="train", categories=categories, shuffle=True, random_state=42
)

# Classes
print(twenty_train.target_names)
print(len(twenty_train.data))

# Visualizando alguns dados (atributos)
print("\n".join(twenty_train.target_names[twenty_train.target[0]]))

# O Scikit-Learn registra os labels como array de números, a fim de aumentar a velocidade
print(twenty_train.target[:10])

# Visualizando as classes dos 10 primeiros registros
for t in twenty_train.target[:10]:
    print(twenty_train.target_names[t])


# Contruindo o Bag of Words (Saco de Palavras)
# Tokenizing
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
count_vect.vocabulary_.get("algorithm")
print(X_train_counts.shape)

# De ocorrências a frequências - Term Frequency times Inverse Document Frequency (Tfidf)
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
print(X_train_tf.shape)

# Mesmo resultado da célular anterior, mas combinando as funções
tfidf_tranformer = TfidfTransformer()
X_train_tfidf = tfidf_tranformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

# Criando modelo Multinomial
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

# Previsões
docs_new = ["God is love", "openGL on the GPU is fast"]
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_tranformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print("%r => %s" % (doc, twenty_train.target_names[category]))

# Criando um Pipeline - Classificador Composto
# vectorizer => tranformer => classifier
text_clf = Pipeline(
    [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", MultinomialNB()),
    ]
)

# Fit
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

# Acurácia do Modelo
twenty_test = fetch_20newsgroups(
    subset="test", categories=categories, shuffle=True, random_state=42
)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target))

# Métricas
print(
    metrics.classification_report(
        twenty_test.target, predicted, target_names=twenty_test.target_names
    )
)

# Confussion Matrix
metrics.confusion_matrix(twenty_test.target, predicted)

# Parâmetros para o GridSearchCV
parameters = {
    "vect__ngram_range": [(1, 1), (1, 2)],
    "tfidf__use_idf": (True, False),
    "clf__alpha": (1e-2, 1e-3),
}

# GridSearchCV
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)

# Fit
gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])

# Teste
print(twenty_train.target_names[gs_clf.predict(["God is love"])[0]])

# Score
print(gs_clf.best_score_)

# Parâmetros utilizados
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
