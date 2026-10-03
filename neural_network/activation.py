import numpy as np
from enum import Enum

class Activation(Enum):
    LINEAR = 0
    RELU = 1
    SIGMOID = 2
    TANH = 3

    def compute(self, x: float):
        match self:
            case Activation.LINEAR:
                return x
            case Activation.RELU:
                return np.maximum(0, x)
            case Activation.SIGMOID:
                return 1 / (1 + np.exp(-x))
            case Activation.TANH:
                return np.tanh(x)
            case _:
                raise ValueError(f"Unsupported activation: {self}")

    def derivative(self, x: float | np.ndarray):
        match self:
            case Activation.LINEAR:
                return np.ones_like(x)
            case Activation.RELU:
                return (x > 0).astype(float)
            case Activation.SIGMOID:
                s = self.compute(x)
                return s * (1 - s)
            case Activation.TANH:
                return 1 - (np.tanh(x) ** 2)
            case _:
                raise ValueError(f"Unsupported activation: {self}")
