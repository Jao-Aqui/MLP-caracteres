import math


class Sigmoide:
    @staticmethod
    def f(x):
        x = max(-700, min(700, x))
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def df(x):
        s = Sigmoide.f(x)
        return s * (1 - s)

def derivada_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return max(0, x)


def derivada_relu(x):
    if x > 0:
        return 1
    return 0


def tanh(x):
    return math.tanh(x)


def derivada_tanh(x):
    t = tanh(x)
    return 1 - t**2

def step(x):
    if x>0: return 1
    if x==0: return 0
    if x<0: return 0
