import nnfs
import numpy as np
from nnfs.datasets import spiral_data

np.random.seed(0)
nnfs.init()


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(100, 3)

layer1 = LayerDense(2, 5)
activation1 = ActivationReLU()
layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output)
