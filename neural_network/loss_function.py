import numpy as np
from enum import Enum

EPSILON = 1e-15

class LossFunction(Enum):
    MSE = 0
    LOG_LOSS = 1

    def compute(self, y_true: float | np.ndarray, y_pred: float | np.ndarray):
        match self:
            case LossFunction.MSE:
                return (y_true - y_pred) ** 2
            case LossFunction.LOG_LOSS:
                y_pred = np.clip(y_pred, EPSILON, 1 - EPSILON)
                return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
            case _:
                raise ValueError(f"Unsupported loss function: {self}")

    def derivative(self, y_true: float | np.ndarray, y_pred: float | np.ndarray):
        match self:
            case LossFunction.MSE:
                return 2 * (y_pred - y_true)
            case LossFunction.LOG_LOSS:
                y_pred = np.clip(y_pred, EPSILON, 1 - EPSILON)
                return (y_pred - y_true) / (y_pred * (1 - y_pred))
            case _:
                raise ValueError(f"Unsupported loss function: {self}")
