import homemade_tensors as ht
import numpy as np

class LinearLayer:
    def __init__(self, in_features, out_features):
        self.weights = ht.Tensor(np.random.randn(1, in_features, out_features) * 1e-3)
        self.bias = ht.Tensor(np.random.randn(1, 1, out_features) * 1e-3)
        
    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, x):
        return ht.MatMul()(x, self.weights) + self.bias

    def update_parameters(self, lr):
        self.weights.data -= np.sum(self.weights.grad, axis=0) * lr
        self.bias.data -= np.sum(self.bias.grad, axis=0) * lr

class MultilayerPerceptron:
    def __init__(self):
        self.l1 = LinearLayer(784, 32)
        self.r1 = ht.ReLU()
        self.l2 = LinearLayer(32, 10)

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.r1(x1)
        x3 = self.l2(x2)
        return x3

    def update_parameters(self, lr):
        self.l1.update_parameters(lr)
        self.l2.update_parameters(lr)