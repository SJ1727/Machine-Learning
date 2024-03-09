import HMDL.Tensors as ht
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

class ReLUMultilayerPerceptron:
    def __init__(self, layer_sizes):
        self._layers = [LinearLayer(in_size, out_size) for in_size, out_size in zip(layer_sizes, layer_sizes[1:])]
        self._parameters = tuple(parameter for layer in self._layers for parameter in layer.parameters())

    def forward(self, x):
        for layer in self._layers:
            x = layer.forward(x)

            if layer is not self._layers[-1]:
                x = ht.ReLU()(x)

        return x

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

class MSELoss:
    def __init__(self):
        self.a = None
        self.actual = None

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, a, actual):
        self.a = a
        self.actual = actual
        error = self.a - self.actual
        self.out = ht.Tensor(
            (error * error).data / self.actual.data.shape[-1],
            _children=(a,)
        )
        self.out.grad_fn = self._backward
        return ht.Sum()(self.out)

    def _backward(self):
        self.a.grad = 2 / self.actual.data.shape[-1] * (self.a.data - self.actual)