import numpy as np
from math_helpers import *

# We Will be Implementing these 4 Functions Today
class LinearNeuralNetwork:
    def __init__(self, layer_sizes:list):
        #  layer_size = [5, 3, 4, 6]
        self.weights = []
        self.biases = []
        self.num_layers = len(layer_sizes) - 1

        for index in range(len(layer_sizes)-1):
            weight = np.random.normal(loc = 0, scale = 1, size =(layer_sizes[index], layer_sizes[index+1]))
            bias = np.zeros(layer_sizes[index+1])

            self.weights.append(weight)
            self.biases.append(bias)

        ...

    def forward(self, x_input):
        current_activation = x_input
        activations = [x_input]

        for weights, biases, layer in zip(self.weights, self.biases, range(self.num_layers)):
            if layer!=0:
                current = sigmoid(current_activation@weights + biases)
            else:
                current = current_activation@weights + biases
            activations.append(current)
            current_activation = current

        return activations
        ...

    def backward(self, y_true, post_activations, lr = 0.5):
        n = y_true.shape[0] if y_true.ndim > 0 else 1
        error = mse_derivative(y_true, post_activations[-1]) /n
        current_activation_derivative = error * sigmoid_derivative_A(post_activations[-1])
        
        for layer in reversed(range(self.num_layers)):
            a_prev = post_activations[layer]
            weights = self.weights[layer]

            dz_dw = a_prev.T @ current_activation_derivative
            dz_db = np.sum(current_activation_derivative, axis=0)

            if layer > 0:
                dz_da_prev = current_activation_derivative @ weights.T
                if layer == 1:
                    # Layer 0 was linear, so derivative of linear activation is 1
                    current_activation_derivative = dz_da_prev
                else:
                    current_activation_derivative = dz_da_prev * sigmoid_derivative_A(post_activations[layer])

            self.weights[layer] -= lr * dz_dw
            self.biases[layer] -= lr * dz_db    
        ...

    def train(self, x_in, y_in, epochs=10, lr=0.1):
        for epoch in range(epochs):
            forward = self.forward(x_in)

            y_pred = forward[-1]
            loss = mse(y_in, y_pred)

            self.backward(y_in, forward, lr=lr)

            print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")
        ...

layer_sizes = [5,3,4, 6]
net = LinearNeuralNetwork(layer_sizes)

print(net.forward([1,2,3,4,5]))