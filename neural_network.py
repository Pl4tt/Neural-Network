import numpy as np
import math
from typing import Iterable


class ActivationFunction:
    @staticmethod
    def forward(x: int | float | Iterable) -> float | np.ndarray: pass

    @staticmethod
    def derivative(x: int | float | Iterable) -> float | np.ndarray: pass

class Sigmoid(ActivationFunction):
    @staticmethod
    def forward(x):
        if type(x) == float or type(x) == int:
            return 1/(1+math.exp(-x))

        return np.array(list(map(lambda element: Sigmoid.forward(element), x)))

    @staticmethod
    def derivative(x):
        if type(x) == float or type(x) == int:
            return Sigmoid.forward(x)*(1-Sigmoid.forward(x))

        return np.array(list(map(lambda element: Sigmoid.derivative(element), x)))

class ReLU(ActivationFunction):
    @staticmethod
    def forward(x):
        if type(x) == float or type(x) == int:
            return x if x > 0 else 0

        return np.array(list(map(lambda element: ReLU.forward(element), x)))

    @staticmethod
    def derivative(x):
        if type(x) == float or type(x) == int:
            return 1 if x > 0 else 0

        return np.array(list(map(lambda element: ReLU.derivative(element), x)))

class Linear(ActivationFunction):
    @staticmethod
    def forward(x):
        if type(x) == float or type(x) == int:
            return x

        return np.array(list(map(lambda element: Linear.forward(element), x)))

    @staticmethod
    def derivative(x):
        if type(x) == float or type(x) == int:
            return 1

        return np.array(list(map(lambda element: Linear.derivative(element), x)))

class Tanh(ActivationFunction):
    @staticmethod
    def forward(x):
        if type(x) == float or type(x) == int:
            return math.tanh(x)

        return np.array(list(map(lambda element: Tanh.forward(element), x)))

    @staticmethod
    def derivative(x):
        if type(x) == float or type(x) == int:
            return 1 - Tanh.forward(x)**2

        return np.array(list(map(lambda element: Tanh.derivative(element), x)))


class CostFunction:
    @staticmethod
    def forward(predicted_y: np.ndarray, actual_y: np.ndarray) -> float: pass

    @staticmethod
    def derivative(predicted_y: np.ndarray, actual_y: np.ndarray) -> np.ndarray: pass

class MeanSquaredError(CostFunction):
    @staticmethod
    def forward(predicted_y, actual_y):
        return np.sum(np.square(np.subtract(predicted_y, actual_y)))/2

    @staticmethod
    def derivative(predicted_y, actual_y):
        return np.subtract(predicted_y, actual_y)


class Layer:
    def __init__(self, neuron_count: int, input_shape: tuple[int], activation_function: ActivationFunction) -> None:
        self.neuron_count = neuron_count
        self.input_shape = input_shape
        self.activation_function = activation_function

        self.weights = np.random.rand(self.neuron_count, self.input_shape)
        self.biases = np.random.rand(self.neuron_count)

    def forward(self, input_arr: np.ndarray) -> np.ndarray:
        if self.input_shape != np.shape(input_arr):
            raise ValueError(f"The input shape is supposed to be {self.input_shape}. {np.shape(input_arr)} was given instead.")
        return input_arr

class DenseLayer(Layer):
    def forward(self, input_arr):
        input_arr = super().forward(input_arr)

        output_arr = np.dot(self.weights, input_arr) + self.biases

        return self.activation_function.forward(output_arr)

class InputLayer(Layer):
    def __init__(self, neuron_count: int):
        self.neuron_count = neuron_count
        self.input_shape = (self.neuron_count,)


class NeuralNetwork:
    def __init__(self, layers: Iterable[Layer], cost_function: CostFunction=None) -> None:
        self.layers = layers
        self.layer_count = len(layers)

        if cost_function is not None:
            self.cost_function = cost_function
        else:
            self.cost_function = MeanSquaredError
        
        self.input_shape = self.layers[0].input_shape
        self.output_shape = self.layers[-1].neuron_count
    
    def train_epoch(self, x: np.ndarray, y: np.ndarray, batch_size: int, learning_rate: float) -> None:
        random_array = np.random.randint(0, len(x), size=batch_size)
        baches = x[random_array,:]


    def train(self, x: np.ndarray, y: np.ndarray, batch_size: int=None, learning_rate: float=0.03, epoch_count: int=100) -> None:
        if batch_size is None:
            batch_size = len(x)//epoch_count

        for i in range(epoch_count):
            self.train_epoch(x, y, batch_size, learning_rate)


if __name__ == "__main__":
    neural_network = NeuralNetwork([
        InputLayer(784),
        DenseLayer(16, 784, ReLU),
        DenseLayer(16, 16, Sigmoid),
        DenseLayer(10, 16, Sigmoid)
    ], MeanSquaredError)
    