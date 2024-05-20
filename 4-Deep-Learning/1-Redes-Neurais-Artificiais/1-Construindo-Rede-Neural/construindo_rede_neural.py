# Construindo a Rede Neural com Programação e Matemática

# Teremos 2 Partes:
# - Parte 1 - Vamos construir uma rede neural artificial somente
# com operações matemáticas
# - Parte 2 - Vamos treinar a rede para Prever a Ocorrência de Câncer

# Por enquanto precisaremos somente do NumPy
import numpy as np


# Parte 1 - Implementando Uma Rede Neural Artificial Somente com Fórmulas
# Matemáticas (Sem Frameworks)

# Parte 1A - Forward Propagation


# Desenvolvendo a Função Para Inicialização de Pesos


# Função para inicialização randômica dos parâmetros do modelo
def inicializa_parametros(dims_camada_entrada):

    # Dicionário para os parâmetros
    parameters = {}

    # Comprimento das dimensões das camadas
    comp = len(dims_camada_entrada)

    # Loop pelo comprimento
    for i in range(1, comp):

        # Inicialização da matriz de pesos
        parameters["W" + str(i)] = (
            np.random.randn(dims_camada_entrada[i], dims_camada_entrada[i - 1]) * 0.01
        )

        # Inicialização do bias
        parameters["b" + str(i)] = np.zeros((dims_camada_entrada[i], 1))

    return parameters


# Desenvolvendo a Função Sigmóide


# Função sigmóide
def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A, Z


# Desenvolvendo a Função ReLU


# Função de ativação ReLu (Rectified Linear Unit)
def relu(Z):
    A = abs(Z * (Z > 0))
    return A, Z


# Desenvolvendo a Ativação Linear


# Operação de ativação
# A é a matriz com os dados de entrada
# W é a matriz de pesos
# b é o bias
def linear_activation(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


# Construindo o Processo de Forward Propagation


# Movimento para frente (forward)
def forward(A_prev, W, b, activation):

    # Se a função de ativação for Sigmoid, entramos neste bloco
    if activation == "sigmoid":
        Z, linear_cache = linear_activation(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    # Se não, se for ReLu, entramos neste bloco
    elif activation == "relu":
        Z, linear_cache = linear_activation(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache


# Combinando Ativação e Propagação


# Propagação para frente
def forward_propagation(X, parameters):

    # Lista de valores anteriores (cache)
    caches = []

    # Dados de entrada
    A = X

    # Comprimento dos parâmetros
    L = len(parameters) // 2

    # Loop
    for i in range(1, L):

        # Guarda o valor prévio de A
        A_prev = A

        # Executa o forward
        A, cache = forward(
            A_prev,
            parameters["W" + str(i)],
            parameters["b" + str(i)],
            activation="relu",
        )

        # Grava o cache
        caches.append(cache)

    # Saída na última camada
    A_last, cache = forward(
        A, parameters["W" + str(L)], parameters["b" + str(L)], activation="sigmoid"
    )

    # Grava o cache
    caches.append(cache)

    return (A_last, caches)


# Desenvolvendo a Função de Custo


# Função de custo (ou função de erro)
def calcula_custo(A_last, Y):

    # Ajusta o shape de Y para obter seu comprimento (total de elementos)
    m = Y.shape[1]

    # Calcula o custo comparando valor real e previso
    custo = (-1 / m) * np.sum((Y * np.log(A_last)) + ((1 - Y) * np.log(1 - A_last)))

    # Ajusta o shape do custo
    custo = np.squeeze(custo)

    return custo


# Parte 1B - Backward Propagation


# Função Sigmóide Backward


# Função sigmoid para o backpropagation
# Fazemos o cálculo da derivada pois não queremos o valor completo da função, mas sim sua variação
def sigmoid_backward(da, Z):

    # Calculamos a derivada de Z
    dg = (1 / (1 + np.exp(-Z))) * (1 - (1 / (1 + np.exp(-Z))))

    # Encontramos a mudança na derivada de z
    dz = da * dg
    return dz


# Compare com a função sigmoid do forward propagation
# A = 1 / (1 + np.exp(-Z))


# Função ReLu Backward


# Função relu para o backpropagation
# Fazemos o cálculo da derivada pois não queremos o valor completo da função, mas sim sua variação
def relu_backward(da, Z):

    dg = 1 * (Z >= 0)
    dz = da * dg
    return dz


# Compare com a função relu do forward propagation:
# A = abs(Z * (Z > 0))


# Ativação Linear Backward


# Ativação linear para o backpropagation
def linear_backward_function(dz, cache):

    # Recebe os valores do cache (memória)
    A_prev, W, b = cache

    # Shape de m
    m = A_prev.shape[1]

    # Calcula a derivada de W (resultado da operação com dz)
    dW = (1 / m) * np.dot(dz, A_prev.T)

    # Calcula a derivada de b (resultado da operação com dz)
    db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

    # Calcula a derivada da operação
    dA_prev = np.dot(W.T, dz)

    return dA_prev, dW, db


# Função que define o tipo de ativação (relu ou sigmoid)
def linear_activation_backward(dA, cache, activation):

    # Extrai o cache
    linear_cache, activation_cache = cache

    # Verifica se a ativação é relu
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward_function(dZ, linear_cache)

    # Verifica se a ativação é sigmoid
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward_function(dZ, linear_cache)

    return dA_prev, dW, db


# Algoritmo Backpropagation


# Algoritmo Backpropagation (calcula os gradientes para atualização dos pesos)
# AL = Valor previsto no Forward
# Y = Valor real
def backward_propagation(AL, Y, caches):

    # Dicionário para os gradientes
    grads = {}

    # Comprimento dos dados (que estão no cache)
    L = len(caches)

    # Extrai o comprimento para o valor de m
    m = AL.shape[1]

    # Ajusta o shape de Y
    Y = Y.reshape(AL.shape)

    # Calcula a derivada da previsão final da rede (feita ao final do Forward Propagation)
    dAL = -((Y / AL) - ((1 - Y) / (1 - AL)))

    # Captura o valor corrente do cache
    current_cache = caches[L - 1]

    # Gera a lista de gradiente para os dados, os pesos e o bias
    # Fazemos isso uma vez, pois estamos na parte final da rede, iniciando o caminho de volta
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = (
        linear_activation_backward(dAL, current_cache, activation="sigmoid")
    )

    # Loop para calcular a derivada durante as ativações lineares com a relu
    for l in reversed(range(L - 1)):

        # Cache atual
        current_cache = caches[l]

        # Calcula as derivadas
        dA_prev, dW, db = linear_activation_backward(
            grads["dA" + str(l + 1)], current_cache, activation="relu"
        )

        # Alimenta os gradientes na lista, usando o índice respectivo
        grads["dA" + str(l)] = dA_prev
        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db

    return grads


# Gradientes e Atualização dos Pesos


# Função de atualização de pesos
def atualiza_pesos(parameters, grads, learning_rate):

    # Comprimento da estrutura de dados com os parâmetros (pesos e bias)
    L = len(parameters) // 2

    # Loop para atualização dos pesos
    for l in range(L):

        # Atualização dos pesos
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - (
            learning_rate * grads["dW" + str(l + 1)]
        )

        # Atualização do bias
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - (
            learning_rate * grads["db" + str(l + 1)]
        )

    return parameters


# Implementando a Rede Completa


# Modelo completo da rede neural
def modeloNN(X, Y, dims_camada_entrada, learning_rate=0.0075, num_iterations=100):

    # Lista para receber o custo a cada época de treinamento
    custos = []

    # Inicializa os parâmetros
    parametros = inicializa_parametros(dims_camada_entrada)

    # Loop pelo número de iterações (épocas)
    for i in range(num_iterations):

        # Forward Propagation
        AL, caches = forward_propagation(X, parametros)

        # Calcula o custo
        custo = calcula_custo(AL, Y)

        # Backward Propagation
        # Nota: ao invés de AL e Y, poderíamos passar somente o valor do custo
        # Estamos passando o valor de AL e Y para fique claro didaticamente o que está sendo feito
        gradientes = backward_propagation(AL, Y, caches)

        # Atualiza os pesos
        parametros = atualiza_pesos(parametros, gradientes, learning_rate)

        # Print do valor intermediário do custo
        # A redução do custo indica o aprendizado do modelo
        if i % 10 == 0:
            print("Custo Após " + str(i) + " iterações é " + str(custo))
            custos.append(custo)

    return parametros, custos


# Função para fazer as previsões
# Não precisamos do Backpropagation pois ao fazer previsões como o modelo
# treinado, teremos os melhores valores de pesos (parametros)
def predict(X, parametros):
    AL, caches = forward_propagation(X, parametros)
    return AL
