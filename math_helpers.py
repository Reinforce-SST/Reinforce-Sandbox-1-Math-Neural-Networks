import numpy as np

def sigmoid(array:np.array):
    return 1/(1+np.exp(-1*np.clip(array, -500, 500)))

def sigmoid_derivative_Z(array:np.array):
    return sigmoid(array)*(1-sigmoid(array))

def sigmoid_derivative_A(array:np.array):
    return array*(1-array)

def log_loss(y_true, y_pred):
    return -1*np.sum(y_true*np.log(y_pred))

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_derivative(y_true, y_pred):
    return 2*(y_pred-y_true)

def log_loss_derivative(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
    return -(y_true/ y_pred)