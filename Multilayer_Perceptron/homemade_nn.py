import homemade_tensors as ht
import numpy as np
class LinearLayer:
    def __init__(self, in_features, out_features):
        self.weights = ht.Tensor(np.random.randn(in_features, out_features) * 1e-3)
        self.bias = ht.Tensor(np.random.randn(1, out_features) * 1e-3)

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, x):
        return ht.MatMul()(x, self.weights) + self.bias

    def parameters(self):
        return (self.weights, self.bias)

class MultilayerPerceptron:
    def __init__(self):
        self.l1 = LinearLayer(784, 32)
        self.r1 = ht.ReLU()
        self.l2 = LinearLayer(32, 10)
        self._parameters = (*self.l1.parameters(), *self.l2.parameters())

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.r1(x1)
        x3 = self.l2(x2)
        return x3

    def parameters(self):
        return self._parameters

class CrossEntropyLoss:
    def __init__(self):
        self.a = None
        self.actual = None
        self.softmax = ht.SoftMax()

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, a, actual):
        self.a = a
        self.actual = actual
        s_a = self.softmax(a)
        self.out = ht.Tensor(
            (actual * ht.Log()(s_a)).data,
            _children=(a,)
        )
        self.out.grad_fn = self._backward
        return ht.Sum()(self.out)

    def _backward(self):
        self.a.grad = self.a.data - self.actual.data

class StochasticGradientDescent:
    def __init__(self, parameters, lr=1e-3):
        self._parameters = parameters
        self.lr = lr

    def step(self):
        for parameter in self._parameters:
            parameter.data -= np.sum(parameter.grad, axis=0) * self.lr