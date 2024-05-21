# Classificação de Imagens com Deep Learning e PyTorch

# Imports
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow
import warnings

warnings.filterwarnings("ignore")

# Verificando a GPU

# Verifica se a plataforma CUDA está disponível
train_on_gpu = torch.cuda.is_available()

# Mensagem para o usuário
if not train_on_gpu:
    print(
        "Plataforma CUDA não está disponível. O treinamento será realizado com a CPU ..."
    )
else:
    print("Plataforma CUDA está disponível! O treinamento será realizado com a GPU ...")


# Carregando o Dataset

# Função que converte os dados em um tensor normalizado
transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Download dos dados de treino
dados_treino = datasets.CIFAR10("dados", train=True, download=True, transform=transform)

# Download dos dados de teste
dados_teste = datasets.CIFAR10("dados", train=False, download=True, transform=transform)


# Preparando os Data Loaders

# Dados de treino
print("Dados de treino:\n", dados_treino)

# Dados de teste
print("Dados de teste:\n", dados_teste)

# Número de amostras de treino
num_amostras_treino = len(dados_treino)
print("Número de amostras: ", num_amostras_treino)

# Criamos um índice e o tornamos randômico
indices = list(range(num_amostras_treino))
np.random.shuffle(indices)

# Percentual dos dados de treino que usaremos no dataset de validação
valid_size = 0.2

# Agora fazemos o split para os dados de treino e validação
split = int(np.floor(valid_size * num_amostras_treino))
idx_treino, idx_valid = indices[split:], indices[:split]

# Definimos as amostras de treino
amostras_treino = SubsetRandomSampler(idx_treino)

# Definimos as amostras de validação
amostras_valid = SubsetRandomSampler(idx_valid)

# Número de subprocessos para carregar os dados
num_workers = 0

# Número de amostras por batch
batch_size = 20

# Data Loader de dados de treino
loader_treino = torch.utils.data.DataLoader(
    dados_treino,
    batch_size=batch_size,
    sampler=amostras_treino,
    num_workers=num_workers,
)

# Data Loader de dados de validação
loader_valid = torch.utils.data.DataLoader(
    dados_treino, batch_size=batch_size, sampler=amostras_valid, num_workers=num_workers
)

# Data Loader de dados de teste
loader_teste = torch.utils.data.DataLoader(
    dados_teste, batch_size=batch_size, num_workers=num_workers
)

# Lista de classes das imagens
classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# Visualizando os Dados


# Função para desnormalização das imagens
def imshow(img):

    # Desfaz a normalização
    img = img / 2 + 0.5

    # Converte em tensor e imprime
    plt.imshow(np.transpose(img, (1, 2, 0)))


# Obtém um batch de dados de treino
dataiter = iter(loader_treino)
images, labels = dataiter.next()

# Converte as imagens em formato NumPy
images = images.numpy()

# Plot de um batch de imagens de treino

# Área de plotagem
fig = plt.figure(figsize=(25, 4))

# Loop e print
for idx in np.arange(20):

    # Cria os subplots
    ax = fig.add_subplot(2, 20 / 2, idx + 1, xticks=[], yticks=[])

    # Desfaz a normalização
    # images[idx]
    imshow(images[idx])

    # Coloca o título
    ax.set_title(classes[labels[idx]])


# Visualizando Uma Imagem em Mais Detalhes

# Extraímos os canais de cores
rgb_img = np.squeeze(images[3])
channels = ["Canal Vermelho (Red)", "Canal Verde (Green)", "Canal Azul (Blue)"]

# Loop e print

# Área de plotagem
fig = plt.figure(figsize=(36, 36))

# Loop pelas imagens
for idx in np.arange(rgb_img.shape[0]):

    # Subplot
    ax = fig.add_subplot(1, 3, idx + 1)

    # Índice
    img = rgb_img[idx]

    # Mostra a imagem em escala de cinza
    ax.imshow(img, cmap="gray")

    # Título
    ax.set_title(channels[idx])

    # Largura e altura da imagem
    width, height = img.shape

    # Limite
    thresh = img.max() / 2.5

    # Loop
    for x in range(width):
        for y in range(height):
            val = round(img[x][y], 2) if img[x][y] != 0 else 0
            ax.annotate(
                str(val),
                xy=(y, x),
                horizontalalignment="center",
                verticalalignment="center",
                size=8,
                color="white" if img[x][y] < thresh else "black",
            )


# Definindo a Arquitetura da Rede


# Arquitetura do Modelo
class ModeloCNN(nn.Module):

    # Método construtor
    def __init__(self):
        super(ModeloCNN, self).__init__()

        # Camada Convolucional de entrada
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)

        # Camada Convolucional oculta
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        # Camada Convolucional oculta
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        # Camada de Max Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Camada Totalmente Conectada 1
        self.fc1 = nn.Linear(64 * 4 * 4, 500)

        # Camada Totalmente Conectada 2
        self.fc2 = nn.Linear(500, 10)

        # Camada de Dropout (Regularização)
        self.dropout = nn.Dropout(0.5)

    # Método Forward
    def forward(self, x):

        # Adiciona uma camada de ativação Relu para cada camada convolucional
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Faz o "achatamento" da matriz resultante da convolução e cria um vetor
        x = x.view(-1, 64 * 4 * 4)

        # Adiciona uma camada de dropout para regularização
        x = self.dropout(x)

        # Adiciona a 1ª camada oculta, com função de ativação relu
        x = F.relu(self.fc1(x))

        # Adiciona uma camada de dropout para regularização
        x = self.dropout(x)

        # Adiciona a 2ª camada oculta (classificação feita pelo modelo)
        x = self.fc2(x)
        return x


# Cria o modelo
modelo = ModeloCNN()
print(modelo)

# Movemos o modelo para a GPU se disponível
if train_on_gpu:
    modelo.cuda()


# Função de Perda (Loss Function)

# Loss function como categorical cross-entropy
criterion = nn.CrossEntropyLoss()


# Otimizador

# Hiperparâmetro
taxa_aprendizado = 0.01

# Otimizador com SGD
optimizer = optim.SGD(modelo.parameters(), lr=taxa_aprendizado)


# Treinamento

# Número de épocas para treinar o modelo
num_epochs = 30

# hiperparâmetro para controlar a mudança do erro em validação
erro_valid_min = np.Inf

for epoch in range(1, num_epochs + 1):

    # Parâmetros para acompanhar o erro total em treinamento e validação
    erro_treino = 0.0
    erro_valid = 0.0

    # Inicia o treinamento do modelo
    modelo.train()

    # Loop pelos batches de dados de treino
    for batch_idx, (data, target) in enumerate(loader_treino):

        # Move os tensores para a GPU se disponível
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        # Limpa os gradientes de todas as variáveis otimizadas
        optimizer.zero_grad()

        # Forward: calcula as saídas previstas
        output = modelo(data)

        # Calcula o erro no batch
        loss = criterion(output, target)

        # Backward: calcula o gradiente da perda em relação aos parâmetros do modelo
        loss.backward()

        # Realiza uma única etapa de otimização (atualização dos parâmetros)
        optimizer.step()

        # Atualiza o erro total em treino
        erro_treino += loss.item() * data.size(0)

    # Inicia a validação do modelo
    modelo.eval()

    # Loop pelos batches de dados de validação
    for batch_idx, (data, target) in enumerate(loader_valid):

        # Move os tensores para a GPU se disponível
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        # Forward: calcula as saídas previstas
        output = modelo(data)

        # Calcula o erro no batch
        loss = criterion(output, target)

        # Atualiza o erro total de validação
        erro_valid += loss.item() * data.size(0)

    # Calcula o erro médio
    erro_treino = erro_treino / len(loader_treino.dataset)
    erro_valid = erro_valid / len(loader_valid.dataset)

    # Print
    print(
        "\nEpoch: {} \tErro em Treinamento: {:.6f} \tErro em Validação: {:.6f}".format(
            epoch, erro_treino, erro_valid
        )
    )

    # Salva o modelo sempre que a perda em validação diminuir
    if erro_valid <= erro_valid_min:
        print(
            "Erro em Validação foi Reduzido ({:.6f} --> {:.6f}). Salvando o modelo...".format(
                erro_valid_min, erro_valid
            )
        )
        torch.save(modelo.state_dict(), "modelos/modelo_final.pt")
        erro_valid_min = erro_valid


# Carrega o Modelo Final

# Carrega o modelo
modelo.load_state_dict(torch.load("modelos/modelo_final.pt"))


# Testando e Avaliando o Modelo Final

# Erro em teste
erro_teste = 0.0

# Controle de acertos do modelo
class_correct = list(0.0 for i in range(10))
class_total = list(0.0 for i in range(10))

# Inicia a avaliação do modelo
modelo.eval()

# Loop pelos batches de dados de teste
for batch_idx, (data, target) in enumerate(loader_teste):

    # Move os tensores para GPU se disponível
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()

    # Forward
    output = modelo(data)

    # Calcula o erro
    loss = criterion(output, target)

    # Atualiza o erro em teste
    erro_teste += loss.item() * data.size(0)

    # Converte probabilidades de saída em classe prevista
    _, pred = torch.max(output, 1)

    # Compara as previsões com o rótulo verdadeiro
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = (
        np.squeeze(correct_tensor.numpy())
        if not train_on_gpu
        else np.squeeze(correct_tensor.cpu().numpy())
    )

    # Calcula a precisão do teste para cada classe
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# Erro médio em teste
erro_teste = erro_teste / len(loader_teste.dataset)
print("\nErro em Teste: {:.6f}\n".format(erro_teste))

# Calcula a acurácia para cada classe
for i in range(10):
    if class_total[i] > 0:
        print(
            "Acurácia em Teste da classe %5s: %2d%% (%2d/%2d)"
            % (
                classes[i],
                100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]),
                np.sum(class_total[i]),
            )
        )
    else:
        print("Acurácia em Teste de %5s:)" % (classes[i]))

# Calcula a acurácia total
print(
    "\nAcurácia em Teste (Total): %2d%% (%2d/%2d)"
    % (
        100.0 * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct),
        np.sum(class_total),
    )
)


# Previsões com o Modelo Treinado

# Obtém um batch de dados de teste
dataiter = iter(loader_teste)
images, labels = dataiter.next()
images.numpy()

# Move as imagens para a GPU se disponível
if train_on_gpu:
    images = images.cuda()

# Faz as previsões com o modelo treinado
output = modelo(images)

# Converte probabilidades de saída em classe prevista
_, preds_tensor = torch.max(output, 1)
preds = (
    np.squeeze(preds_tensor.numpy())
    if not train_on_gpu
    else np.squeeze(preds_tensor.cpu().numpy())
)

# Plot das previsões
fig = plt.figure(figsize=(25, 4))
print("\nEntre parênteses a classe real. Vermelho indica erro do modelo.\n")
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20 / 2, idx + 1, xticks=[], yticks=[])
    imshow(images[idx].cpu())
    ax.set_title(
        "{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
        color=("green" if preds[idx] == labels[idx].item() else "red"),
    )
