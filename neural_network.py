import numpy as np
import time
from typing import Iterable
from mnist import MNIST


class ActivationFunction:
    @staticmethod
    def forward(x: int | float | Iterable) -> float | np.ndarray: pass

    @staticmethod
    def derivative(x: int | float | Iterable) -> float | np.ndarray: pass

class Sigmoid(ActivationFunction):
    @staticmethod
    def forward(x):
        return 1/(1+np.exp(-x))

    @staticmethod
    def derivative(x):
        return Sigmoid.forward(x)*(1-Sigmoid.forward(x))

class ReLU(ActivationFunction):
    @staticmethod
    def forward(x):
        if type(x) == float or type(x) == int or type(x) == np.float64:
            return x if x > 0 else 0

        return np.array(list(map(lambda element: ReLU.forward(element), x)))

    @staticmethod
    def derivative(x):
        if type(x) == float or type(x) == int or type(x) == np.float64:
            return 1 if x > 0 else 0

        return np.array(list(map(lambda element: ReLU.derivative(element), x)))

class Linear(ActivationFunction):
    @staticmethod
    def forward(x):
        return x

    @staticmethod
    def derivative(x):
        return np.ones(np.shape(x))

class Tanh(ActivationFunction):
    @staticmethod
    def forward(x):
        return np.tanh(x)

    @staticmethod
    def derivative(x):
        return 1 - Tanh.forward(x)**2


class CostFunction:
    @staticmethod
    def forward(predicted_y: np.ndarray, actual_y: np.ndarray) -> float | int: pass

    @staticmethod
    def derivative(predicted_y: np.ndarray, actual_y: np.ndarray) -> np.ndarray: pass

class MeanSquaredError(CostFunction):
    @staticmethod
    def forward(predicted_y, actual_y):
        return np.sum(np.square(predicted_y - actual_y))/2

    @staticmethod
    def derivative(predicted_y, actual_y):
        return predicted_y - actual_y


class Layer:
    def __init__(self, neuron_count: int, input_shape: tuple[int], activation_function: ActivationFunction) -> None:
        self.neuron_count = neuron_count
        self.input_shape = input_shape
        self.activation_function = activation_function

        # self.weights = np.random.rand(self.neuron_count, *self.input_shape)
        # self.biases = np.random.rand(self.neuron_count)
        self.weights = np.random.uniform(0, 0.0001, size=(self.neuron_count, *self.input_shape))
        self.biases = np.random.uniform(0, 0.0001, size=(self.neuron_count,))
        
        # print("W", self.weights)
        # print("B", self.biases)
        self.set_update_arrays()
    
    def set_update_arrays(self) -> None:
        self.update_weights = np.zeros((self.neuron_count, *self.input_shape))
        self.update_biases = np.zeros((self.neuron_count,))

    def forward(self, input_arr: np.ndarray, mode: int=1) -> np.ndarray:
        if self.input_shape != np.shape(input_arr):
            raise ValueError(f"The input shape is supposed to be {self.input_shape}. {np.shape(input_arr)} was given instead.")
        return input_arr
    
    def update_values(self, learning_rate: float) -> None:
        self.weights -= learning_rate*self.update_weights
        self.biases -= learning_rate*self.update_biases
        # print(self.update_weights, self.update_biases)
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

    def train_batch_sgd(self, mini_batch: np.ndarray, learning_rate: float) -> None:
        for x, y in mini_batch:  # backpropagation / sgd
            useful_y = np.zeros((10,))
            useful_y[y-1] = 1

            activation = x
            activations = [x]
            zs = []

            for layer in self.layers:  # forward pass
                z = layer.forward(activation, mode=0)
                zs.append(z)
                activation = layer.forward(activation)
                activations.append(activation)

            # backward pass
            # print("zs", zs[-1])
            # print("D", self.cost_function.derivative(activations[-1], useful_y), self.layers[-1].activation_function.derivative(zs[-1]))
            delta = self.cost_function.derivative(activations[-1], useful_y)*self.layers[-1].activation_function.derivative(zs[-1])
            # print("B", delta, "W", np.atleast_2d(delta).T@np.atleast_2d(activations[-2]))
            self.layers[-1].update_biases += delta
            self.layers[-1].update_weights += np.atleast_2d(delta).T@np.atleast_2d(activations[-2])

            for layer in range(2, len(self.layers)):
                # print("zs", zs[-layer])
                a_derivative = self.layers[-layer].activation_function.derivative(zs[-layer])
                # print("F", self.layers[-layer+1].weights.T@delta, "S", a_derivative)
                delta = (self.layers[-layer+1].weights.T@delta)*a_derivative
                # print("B", delta, "W", np.atleast_2d(delta).T@np.atleast_2d(activations[-layer-1]))
                self.layers[-layer].update_biases += delta
                self.layers[-layer].update_weights += np.atleast_2d(delta).T@np.atleast_2d(activations[-layer-1])

        for layer in self.layers[1:]:  # updateing weights and biases
            layer.update_biases /= len(mini_batch)
            layer.update_weights /= len(mini_batch)
            layer.update_values(learning_rate)

    def train_epoch(self, datasets: np.ndarray, batch_size: int, learning_rate: float) -> None:
        random_array = np.random.randint(0, len(datasets), size=batch_size)
        mini_batch = datasets[random_array,:]

        self.train_batch_sgd(mini_batch, learning_rate)

    def train(self, x: np.ndarray, y: np.ndarray, test: tuple[np.ndarray], batch_size: int=None, learning_rate: float=0.03, epoch_count: int=100) -> None:
        if batch_size is None:
            batch_size = len(x)//4
        
        datasets = np.array(list(zip(x, y)), dtype=object)

        for i in range(epoch_count):
            self.train_epoch(datasets, batch_size, learning_rate)

            print(f"Epoch {i+1} finished")
            # print(self.predict(test[0]), test[1])
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        result = x
        # print(x)
        for layer in self.layers:
            result = layer.forward(result)
            # print(result)

        return np.array(result)


if __name__ == "__main__":
    neural_network = NeuralNetwork([
        InputLayer(784),
        DenseLayer(500, (784,), ReLU),
        DenseLayer(200, (500,), Sigmoid),
        DenseLayer(100, (200,), Linear),
        DenseLayer(10, (100,), Sigmoid),
    ], MeanSquaredError)


    mndata = MNIST("digit_data")

    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    train_images = mndata.process_images_to_numpy(train_images)
    train_labels = mndata.process_images_to_numpy(train_labels)
    test_images = mndata.process_images_to_numpy(test_images)
    test_labels = mndata.process_images_to_numpy(test_labels)
    
    prediction = neural_network.predict(np.array(train_images[0]))

    neural_network.train(train_images, train_labels, (test_images[0], test_labels[0]), batch_size=1000, epoch_count=40)
    # neural_network.predict(test_images[1])
    # print(test_labels[1])
    for image, label in zip(test_images[:10], test_labels[:10]):
        prediction = neural_network.predict(image)
        print(prediction, np.argmax(prediction), label)
    
    # for layer in neural_network.layers[1:]:
    #     print(layer.weights, layer.biases)