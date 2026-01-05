import math


def matmul(n, m): # TODO
    pass

def mid_value(x):
    return sum(x) / len(x)

def disp(x):
    _mid = mid_value(x)
    return sum([(_mid - i)**2 for i in x]) / len(x)

def norm(x):
    _mid = mid_value(x)
    _disp = disp(x)
    return [(i - _mid) / math.sqrt(_disp + 1e-5) for i in x]

def layer_norm(x, weight: float = 1.0, bias: float = 0):
    n = len(x)
    mu = sum(x) / n # count middle value
    
    variance = sum((i - mu)**2 for i in x) / n # its dispersion
    
    eps = 1e-5
    std_inv = 1.0 / math.sqrt(variance + eps)

    return [((i - mu) * std_inv) * weight + bias for i in x]


class LayerNorm:
    def __init__(self, features, eps=1e-5) -> None: # features it's a count of neurons
        self.eps = eps

        self.weight = [1.0] * features
        self.bias = [0.0] * features

    def __call__(self, x: tuple | list):
        n = len(x)

        mu = sum(x) / n

        variance = sum((i - mu)**2 for i in x) / n # it's dispersion
        std_inv = 1.0 / math.sqrt(variance + self.eps)
        output = []
        for i in range(n):
            normalized = (x[i] - mu) * std_inv
            res = normalized * self.weight[i] + self.bias[i]
            output.append(res)
        return output
    
    def __repr__(self) -> str:
        return f'LayerNorm(features={len(self.weight)})'


def gelu(x: tuple | list): 
    return [0.5 * i * (1 + math.tanh(math.sqrt(2/math.pi) * (i + 0.044715 * i ** 3))) for i in x]

def normalized_exponential_function(x: tuple | list): # x it's a vector
    _n = tuple(math.exp(1)**i for i in x)
    _sum = sum(_n)
    return [i / _sum for i in _n]


# res = layer_norm([2, 4, 6, 8])
res = LayerNorm(4)
x = [2, 4, 6, 8]
res = res(x)
print('res: ', res)
print('gelu: ', gelu(x))
print('normalized_exponential_function: ', normalized_exponential_function(x))
# print('disp: ', disp(res))
# print('norm: ', norm(res))
