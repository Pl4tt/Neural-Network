import numpy as np


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
