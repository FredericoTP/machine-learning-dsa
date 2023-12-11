# Carregando pacotes
install.packages("caret")
install.packages("ROCR")
install.packages("e1071")

# Carregando o dataset em um dataframe
credito_dataset <- read.csv("credit_dataset_final.csv", header = TRUE, sep = ",")
head(credito_dataset)
summary(credito_dataset)
str(credito_dataset)
View(credito_dataset)

##### Pré-processamento #####

# Transformando variáveis em fatores
to.factors <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- as.factor(df[[variable]])
  }
  return(df)
}

# Normalização
scale.features <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- scale(df[[variable]], center = T, scale = T)
  }
  return(df)
}

# Normalizando as variáveis
numeric.vars <- c("credit.duration.months", "age", "credit.amount")
credito_dataset_scaled <- scale.features(credito_dataset, numeric.vars)

# Variáveis do tipo fator
categorical.vars <- c('credit.rating', 'account.balance', ''previous.credit.payment.status, 'credit.purpose', 'savings', 'employment.duration', 'installment.rate', 'marital.status', 'guarantor', 'residence.duration', 'current.assets', 'other.credits', 'apartment.type', 'bank.credits', 'occupation', 'dependents', 'telephone', 'foreign.worker')

# Aplicando as conversões ao dataset
credito_dataset_final <- to.factors(df = credito_dataset_scaled, variables = categorical.vars)
str(credito_dataset_final)
View(credito_dataset_final)

# Preparando os dados de treino e de teste
indexes <- sample(1:nrow(credito_dataset_final), size = 0.6 * nrow(credito_datset_final))
train.data <- credito_dataset_final[indexes,]
test.data <- credito_dataset_final[-indexes,]
class(train.data)
class(test.data)

# Separando os atributos e as classes
test.feature.vars <- test.data[,-1]
test.class.var <- test.data[,1]
class(test.feature.vars)
class(test.class.var)

# Construindo o modelo de regressão logística
formula.init <- "credit.rating ~ ."
formula.init <- as.formula(formula.init)
help(glm)
modelo_v1 <- glm(formula = formula.init, data = train.data, family = "binomial")

# Visualizando os detalhes do modelo
summary(modelo_v1)

# Fazendo previsões e analisando o resultado
View(test.data)
previsoes <- predict(modelo_v1, test.data, type = "response")
previsoes <- round(previsoes)
View(previsoes)

# Confusion Matrix
confusionMatrix(table(data = previsoes, reference = test.class.var), positive = "1")

# Feature Selection
formula <- "credit.rating ~ ."
formula <- as.formula(formula)
control <- trainControl(method = "repeatedcv", number = 10, repeats = 2)
model <- train(formula, data = train.data, method = "glm", trControl = control)
importance <- varimp(model, scale = FALSE)

# Plot
plot(importance)

# Construindo um novo modelo com as variáveis selecionadas
formula.new <- "credit.rating ~ account.balance + credit.purpose + previous.credit.payment.status + savings + credit.duration.months"
formula.new <- as.formula(formula.new)
modelo_v2 <- glm(formula = formula.new, data = train.data, family = "binomial")

# Prevende e avaliando o modelo
previsoes_new <- predict(modelo_v2, test.data, type = "response")
previsoes_new <- round(previsoes_new)

# Confusion Matrix
confusionmatrix(table(data = previsoes_new, reference = test.class.var), positive = "1")


# Avaliando a performance do modelo

# Plot do modelo com melhor acurácia
modelo_final <- modelo_v2
previsoes <- predict(modelo_final, test.features.vars, type = "response")
avaliacao <- prediction(previsoes, test.class.var)

# Função para Plot ROC
plot.roc.curve <- function(predictions, title.text){
  perf <- performance(avaliacao, "tpr", "fpr")
  plot(perf,col = "black",lty = 1, lwd = 2, main = tittle.text, cex.main = 0.6, cex.lab = 0.8, xaxs = "i", yaxs = "i")
  abline(0.1, col = "red")
  auc <- performance(avaliacao,"auc")
  auc <- unlist(slot(auc, "y.values"))
  auc <- round(auc,2)
  legend(0.4,0.4,legend = c(paste0("AUC: ",auc)), cex = 0.6, bty = "n", box.col = "white")
}

# Plot
par(mfrom = c(1, 2))
plot.roc.curve(avaliacao, tittle.text = "Curva ROC")


# Fazendo previsões em novos dados

# Novos dados
account.balance <- c(1, 3, 3, 2)
credit.purpose <- c(4, 2, 3, 2)
previous.credit.payment.status <- c(3, 3, 2, 2)
savings <- c(2, 3, 2, 3)
credit.duration.months <- c(15, 12, 8, 6)

# Cria um dataframe
novo_dataset <- data.frame(account.balance, credit.purpose, previous.credit.payment.status, savings, credit.duration.months)
class(novo_dataset)
View(novo_dataset)

# Separa variáveis explanatórias numéricas e categóricas
new.numeric.vars <- c("credit.duration.months")
new.categorical.vars <- c('account.balance', 'previous.credit.payment.status', 'credit.purpose', 'savings')

# Aplica as transformações
novo_dataset_final <- to.factors(df = novo_datset, variables = new.categorical.vars)
str(novo_datset_final)

novo_datset_final <- scale.features(novo_datset_final, new.numeric.vars)
str(novo_datset_final)

View(novo_datset_final)

# Previsões
?predict
previsoes_novo_client <- predict(modelo_final, newdata = novo_datset_final, type = "response)
round(previsoes_novo_client)