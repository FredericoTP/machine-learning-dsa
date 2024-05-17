# Introdução ao Framework PyTorch

# Imports
import numpy
import torch
import torchvision

# Criando um tensor
x = torch.tensor([1.0, 2.0])

# Visualiza o tensor
print(x)

# Shape
print(x.shape)

# Criando um tensor
t = torch.tensor([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], dtype=torch.float32)

# No PyTorch, temos duas maneiras de obter a forma (shape):
t.size()
t.shape

# O rank de um tensor é igual ao comprimento da forma do tensor
len(t.shape)
len(x.shape)

# O número de elementos dentro de um tensor  é igual ao produto
# dos valores dos componentes da forma
torch.tensor(t.shape).prod()
torch.tensor(x.shape).prod()

# Retornando um elemento de um tensor
z = torch.tensor([[1.0, 2.0], [5.0, 3.0], [0.0, 4.0]])
print(z)

# Shape
print(z.shape)

# Retornamos a primeira linha (índice 0) e segunda coluna (índice 1)
# O retorno é no formato de tensor
print(z[0][1])

# Retornamos a primeira linha (índice 0) e segunda coluna (índice 1)
# O retorno é no formato de escalar (apenas o valor)
print(z[0][1].item())

# Quando criamos tensores com valores randômicos, passamos apenas
# o número de dimensões.
input1 = torch.randn([1, 4, 4, 2])
input2 = torch.randn(1, 4, 4, 2)
print(input1.shape)
print(input2.shape)
print(len(input1.shape))
print(len(input2.shape))
print(input1)
print(input2)


# -------------------------------------------------


# Array NumPy x Tensor PyTorch

# Cria um array NumPy
a = numpy.array(1)

# Cria um tensor PyTorch
b = torch.tensor(1)

# Tipo
print(type(a))

# Tipo
print(type(b))

# Print
print(a)
print(b)

print(type(b))


# -------------------------------------------------


# Operações com Tensores

# Criamos 2 tensores
t1 = torch.tensor(12)
t2 = torch.tensor(4)
print(t1, t2)

# Soma
print(t1 + t2)

# Subtração
print(t1 - t2)

# Multiplicação
print(t1 * t2)

# Divisão
print(t1 // t2)


# Operações com Matrizes

# Matriz (tensor rank 2) de números randômicos
t_rank2 = torch.randn(3, 3)

# Tensor rank 3 de números randômicos
t_rank3 = torch.randn(3, 3, 3)

# Tensor rank 4 de números randômicos
t_rank4 = torch.randn(3, 3, 3, 3)

# Multiplicação entre 2 tensores
A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = torch.tensor([[9, 8, 7], [6, 5, 4], [3, 2, 1]])

resultado1 = A * B

# Resultado
print(resultado1)

resultado2 = torch.matmul(A, B)

# Resultado
print(resultado2)

resultado3 = torch.sum(A * B)

# Resultado
print(resultado3)

# Para multiplicação de matrizes, fazemos assim em PyTorch:
AB1 = A.mm(B)
# ou
AB2 = torch.mm(A, B)
# ou
AB3 = torch.matmul(A, B)
# Ou assim (Python 3.5+)
AB4 = A @ B

print(AB1)
print(AB2)
print(AB3)
print(AB4)

# Multiplicação de matrizes
A @ B

# Usando seed para iniciar 2 tensores com valores randômicos
torch.manual_seed(42)
a = torch.randn(3, 3)
b = torch.randn(3, 3)

# Adição de matrizes
print(torch.add(a, b))

# Subtração de matrizes
print(torch.sub(a, b))

# Multiplicação de matrizes
print(torch.mm(a, b))

# Divisão de matrizes
print(torch.div(a, b))

# Matriz Original
print(a, "\n")

# Matriz Transposta
torch.t(a)
