from typing import Iterable
import numpy as np


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
