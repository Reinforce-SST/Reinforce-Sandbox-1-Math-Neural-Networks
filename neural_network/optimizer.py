import numpy as np
from enum import Enum

class Optimizer(Enum):
    GD = 0
    SGD = 1
    MINI_BATCH = 2

    def step(self, weights: list[np.ndarray], biases: list[np.ndarray], d_weights: list[np.ndarray], d_biases: list[np.ndarray], learning_rate: float):
        match self:
            case Optimizer.GD | Optimizer.SGD | Optimizer.MINI_BATCH:
                for i in range(len(weights)):
                    weights[i] -= learning_rate * d_weights[i]
                    biases[i] -= learning_rate * d_biases[i]
                return weights, biases
            case _:
                raise ValueError(f"Unsupported optimizer: {self}")

    def batch(self, input: list[np.ndarray], target: list[np.ndarray], num_samples: int, batch_size: int):
        permutation = np.random.permutation(num_samples)
        input_shuffled = input[permutation]
        target_shuffled = target[permutation]

        match self:
            case Optimizer.GD:
                return [(input_shuffled, target_shuffled)]
            case Optimizer.SGD:
                return [(input_shuffled[i:i+1], target_shuffled[i:i+1]) for i in range(num_samples)]
            case Optimizer.MINI_BATCH:
                return [(input_shuffled[i:i+batch_size], target_shuffled[i:i+batch_size]) for i in range(0, num_samples, batch_size)]
            case _:
                raise ValueError(f"Unknown optimizer configuration: {self}")
    