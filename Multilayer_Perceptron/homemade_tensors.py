import numpy as np

class Tensor:
    def __init__(self, data, _children=tuple()):
        self.data = np.array(data)
        self.grad = 0
        self.grad_fn = lambda: None
        self._children = set(_children)

    def __repr__(self):
        return f"{self.data}"

    def backward(self):
        # TODO: Change backpropogration to be in topological order
        self.grad_fn()
        for child in self._children:
            child.backward()

    def __add__(self, other):
        return AddTesnors()(self, other)

    def __mul__(self, other):
        return MultiplyTesnor()(self, other)

    def __sub__(self, other):
        return SubtractTensors()(self, other)

class TensorOperation:
    def __init__(self) -> None:
        self.out = None
    
    def __call__(self, *args) -> Tensor:
        return self.forward(*args)

class MultiplyTesnor(TensorOperation):
    def __init__(self) -> None:
        super(MultiplyTesnor, self).__init__()
        self.a = None
        self.b = None

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        self.a = a
        self.b = b
        self.out = Tensor(
            a.data * b.data,
            _children=(a, b)
        )
        self.out.grad_fn = self._backward
        return self.out

    def _backward(self) -> None:
        self.a.grad = self.out.grad * self.b.data
        self.b.grad = self.out.grad * self.a.data

class AddTesnors(TensorOperation):
    def __init__(self) -> None:
        super(AddTesnors, self).__init__()
        self.a = None
        self.b = None

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        self.a = a
        self.b = b
        self.out = Tensor(
            a.data + b.data,
            _children=(a, b)
        )
        self.out.grad_fn = self._backward
        return self.out

    def _backward(self) -> None:
        self.a.grad = self.out.grad
        self.b.grad = self.out.grad

class SubtractTensors(TensorOperation):
    def __init__(self):
        super(SubtractTensors, self).__init__()
        self.a = None
        self.b = None
    
    def forward(self, a, b):
        self.a = a
        self.b = b
        self.out = Tensor(
            self.a.data - self.b.data,
            _children=(a,b)
        )
        self.out.grad_fn = self._backward
        return self.out

    def _backward(self):
        self.a.grad = self.out.grad
        self.b.grad = -self.out.grad

class Log(TensorOperation):
    def __init__(self) -> None:
        super(Log, self).__init__()
        self.a = None
    
    def forward(self, a):
        self.a = a
        self.out = Tensor(
            np.log(a.data),
            _children=(a,)
        )
        self.out.grad_fn = self._backward
        return self.out

    def _backward(self):
        self.a.grad = np.reciprocal(self.a.data) * self.out.grad

class Sum(TensorOperation):
    def __init__(self):
        super(Sum, self).__init__()
        self.a = None
    
    def forward(self, a):
        self.a = a
        self.out = Tensor(
            np.sum(a.data),
            _children=(a,)
        )
        self.out.grad_fn = self._backward
        return self.out

    def _backward(self):
        self.a.grad = self.out.grad

class ReLU(TensorOperation):
    def __init__(self) -> None:
        super(ReLU, self).__init__()
        self.a = None

    def forward(self, a: Tensor) -> Tensor:
        self.a = a
        self.out = Tensor(
            np.vectorize(lambda x: 0 if x < 0 else x)(a.data),
            _children=(a,)
        )
        self.out.grad_fn = self._backward
        return self.out
    
    def _backward(self):
        self.a.grad = np.vectorize(lambda x: 0 if x <= 0 else 1)(self.a.data) * self.out.grad

class MatMul(TensorOperation):
    def __init__(self) -> None:
        super(MatMul, self).__init__()
        self.a = None
        self.b = None

    def forward(self, a, b):
        self.a = a
        self.b = b
        self.out = Tensor(
            np.matmul(a.data, b.data),
            _children=(a, b)
        )
        self.out.grad_fn = self._backward
        return self.out

    def _backward(self):
        # TODO: Check if these are right
        self.a.grad = np.matmul(self.out.grad, np.transpose(self.b.data, axes=(0, 2, 1)), axes=[(1, 2), (1, 2), (1, 2)])
        self.b.grad = np.matmul(np.transpose(self.a.data, axes=(0, 2, 1)), self.out.grad, axes=[(1, 2), (1, 2), (1, 2)])

class SoftMax(TensorOperation):
    def __init__(self):
        super(SoftMax, self).__init__()
        self.a = None

    def forward(self, a):
        self.a = a
        self.out = Tensor(
            np.array(np.exp(a.data) / np.sum(np.exp(a.data), axis=(1, 2))[:, np.newaxis, np.newaxis]),
            _children=(a,)
        )
        self.out.grad_fn = self._backward
        return self.out

    def _backward(self):
        # TODO: CHECK THIS
        jacobian = np.matmul(self.out.data, -self.out.data.T) + np.diag(self.out.data)
        self.a.grad = np.matmul(jacobian, self.out.grad)

class CrossEntropyLoss:
    def __init__(self):
        self.a = None
        self.actual = None
        self.softmax = SoftMax()

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, a, actual):
        self.a = a
        self.actual = actual
        s_a = self.softmax(a)
        self.out = Tensor(
            (actual * Log()(s_a)).data,
            _children=(a,)
        )
        self.out.grad_fn = self._backward
        return Sum()(self.out)

    def _backward(self):
        self.a.grad = self.a.data - self.actual.data
