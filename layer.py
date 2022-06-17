import numpy as np

from activation_function import ActivationFunction


class Layer:
    def __init__(self, neuron_count: int, input_shape: tuple[int], activation_function: ActivationFunction) -> None:
        self.neuron_count = neuron_count
        self.input_shape = input_shape
        self.activation_function = activation_function

        self.weights = np.random.rand(self.neuron_count, *self.input_shape)
        self.biases = np.random.rand(self.neuron_count)
        
        self.set_update_arrays()
    
    def set_update_arrays(self) -> None:
        self.update_weights = np.zeros((self.neuron_count, *self.input_shape), dtype=np.float64)
        self.update_biases = np.zeros((self.neuron_count,), dtype=np.float64)

    def forward(self, input_arr: np.ndarray, mode: int=1) -> np.ndarray:
        if self.input_shape != np.shape(input_arr):
            raise ValueError(f"The input shape is supposed to be {self.input_shape}. {np.shape(input_arr)} was given instead.")
        return input_arr
    
    def update_values(self, learning_rate: float) -> None:
        self.weights -= learning_rate*self.update_weights
        self.biases -= learning_rate*self.update_biases

        self.set_update_arrays()

class DenseLayer(Layer):
    def forward(self, input_arr, mode=1):
        input_arr = super().forward(input_arr, mode)

        output_arr = self.weights@input_arr + self.biases
        
        if mode == 0:
            return output_arr

        return self.activation_function.forward(output_arr)

class InputLayer(Layer):
    def __init__(self, neuron_count: int):
        self.neuron_count = neuron_count
        self.input_shape = (self.neuron_count,)
