# Criando Árvore de Decisão com pacote rpart

# Criando um dataframe
?expand.grid
clima <- expand.grid(
  Tempo = c("Ensolarado", "Nublado", "Chuvoso"),
  Temperatura = c("Quente", "Ameno", "Frio"),
  Humidade = c("Alta", "Normal"),
  Vento = c("Fraco", "Forte")
)

# Visualizando o dataframe
View(clima)

# Variável target
response <- c(1, 19, 4, 31, 16, 2, 11, 23, 35, 6, 24, 15, 18, 36)

# Gerando um vetor do tipo fator
play <- as.factor(c("Não Jogar", "Não Jogar", "Não Jogar", "Jogar", "Jogar", "Jogar", "Jogar", "Jogar", "Jogar", "Jogar", "Não Jogar", "Jogar", "Jogar", "Não Jogar"))

# Dataframe final
tennis <- data.frame(clima[response,] play)

# Carregando o pacote
install.packages("rpart")
library(rpart)

# Criando o modelo
?rpart
?rpart.control
tennis_tree <- rpart(play ~ .,
  data = tennis,
  method = "class",
  parms = list(split = "information"),
  control = rpart.control(minsplit = 1)
)

# Visualizando o ganho de informação para cada atributo
tennis_tree

# Gerando o plot
install.packages("rpart.plot")
library(rpart.plot)

# Plot
?prp
prp(tennis_tree, type = 0, extra = 1, under = TRUE, compress = TRUE)

# Interpretando a Árvore de Decisão

# Para ler os nós da árvore, basta iniciar a partir do nó superior, que corresponde aos dados de treinamento original e, em seguida, começar a ler as regras, Observe que cada nó tem duas derivações: O ramo esquerdo significa que a regras superior é verdadeira e a direita significa que ela é falsa.

# À esquerda da primeira regra, você vê uma regra terminal importante (uma folha terminal), em um círculo, indicando um resultado positivo, Jogar, que você pode ler como jogar tênis = Verdadeiro. Os números sob a folha terminal mostram quatro exemplos afirmando que esta regra é "yes" e zero afirmando "no".

# COnsidere o atributo "Vento" que pode ter os valores "Fraco" ou "Forte". Calcula-se então a entropia para cada um desses valores e depois a diferença entre a entropia do atributo vento e a soma das entropias de cada um dos valores associados ao atributo, multiplicado pela proporção de exemplos particionados de acordo com o valor (separados de um lado os exemplos com Vento = "Fraco" e do outro lado Vento = "Forte").

# Frequentemente, as regras de árvore de decisão não são imediatamente utilizáveis, e você precisa interpretá-las antes de uso. No entanto, eles são claramente inteligíveis (e muito melhor do que um coeficiente de vetores de valores).

# Fazendo previsões com o modelo

# Dados
clima <- expand.grid(
  Tempo = c("Ensolarado", "Nublado", "Chuvoso"),
  Temperatura = c("Quente", "Ameno", "Frio"),
  Humidade = c("Alta", "Normal"),
  Vento = c("Fraco", "Forte")
)

# Vetor para selecionar as linhas
response <- c(2, 20, 3, 33, 17, 4, 5)

# Novos dados
novos_dados <- data.frame(clima[response,])
View(novos_dados)

# Previsões
?predict
predict(tennis_tree, novos_dados)