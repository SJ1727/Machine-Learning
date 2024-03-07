import numpy as np
import HMDL.Tensors

class StochasticGradientDescent:
    def __init__(self, parameters, lr=1e-3):
        self._parameters = parameters
        self.lr = lr

    def step(self):
        for parameter in self._parameters:
            parameter.data -= np.sum(parameter.grad, axis=0) * self.lr

    def zero_grad(self):
        for parameter in self._parameters:
            parameter.grad = 0