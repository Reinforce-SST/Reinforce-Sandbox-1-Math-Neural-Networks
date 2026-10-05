import numpy as np
from math_helpers import *

# We Will be Implementing these 4 Functions Today
class LinearNeuralNetwork:
    def __init__(self, layer_sizes:list):
        self.weights = []
        self.biases = []
        self.num_layers = len(layer_sizes) - 1
        
        for layer in range(self.num_layers):
            weight = np.random.randn(layer_sizes[layer], layer_sizes[layer+1]) * np.sqrt(2. / layer_sizes[layer])
            bias = np.zeros(layer_sizes[layer+1])
            
            self.weights.append(weight)
            self.biases.append(bias)
            pass

        ...

    def forward(self, x_input):
        current_activation = x_input
        post_activations = []

        post_activations.append(x_input)
        
        for weights, biases, layer in zip(self.weights,self.biases, range(len(self.biases))):
            current_activation = current_activation@weights + biases
            if layer !=0:
                current_activation = sigmoid(current_activation)
                post_activations.append(current_activation)
            else:
                post_activations.append(current_activation)
        return post_activations

    def backward(self, y_true, post_activations, lr = 0.5):
        error = mse_derivative(y_true, post_activations[-1],)
        current_activation_derivative = error * sigmoid_derivative_A(post_activations[-1])
        
        for layer in reversed(range(self.num_layers)):
            a_prev = post_activations[layer]
            weights = self.weights[layer]

            dz_dw = a_prev.T @ current_activation_derivative
            dz_db = np.sum(current_activation_derivative, axis=0)

            if layer > 0:
                dz_da_prev = current_activation_derivative @ weights.T
                current_activation_derivative = dz_da_prev * sigmoid_derivative_A(post_activations[layer])

            self.weights[layer] -= lr * dz_dw
            self.biases[layer] -= lr * dz_db
        ...

    def train(self, x_in, y_in, epochs=1000, lr=0.1):
        for epoch in range(epochs):
            post_activations = self.forward(x_in)

            y_pred = post_activations[-1]
            loss = mse(y_true=y_in, y_pred=y_pred)
            
            self.backward(y_in, post_activations, lr)
                    
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")

