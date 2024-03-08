import numpy as np
import HMDL.Tensors

class Optimizer:
    def zero_grad(self):
        for parameter in self._parameters:
            parameter.grad = 0

class StochasticGradientDescent(Optimizer):
    def __init__(self, parameters, lr=1e-3):
        self._parameters = parameters
        self.lr = lr

    def step(self):
        for parameter in self._parameters:
            parameter.data -= np.sum(parameter.grad, axis=0) * self.lr

class Adam(Optimizer):
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-08):
        self._parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self._m = [np.zeros(parameter.data.shape) for parameter in self._parameters]
        self._v = [np.zeros(parameter.data.shape) for parameter in self._parameters]
        self.t = 1

    def step(self):
        for m, v , parameter in zip(self._m, self._v, self._parameters):
            grad = np.sum(parameter.grad, axis=0)
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * grad ** 2
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            parameter.data -= m_hat / (np.sqrt(v_hat) + self.eps) * self.lr

        self.t += 1