"""
Stage 4 Tests: Optimization & Training Loop (train)
Tests end-to-end parameter updates, training stability (no NaNs), and overfitting on a tiny dataset.
"""

import numpy as np
import pytest
from LinearNeuralNetwork import LinearNeuralNetwork
import math_helpers


def test_train_weight_update():
    """Run a single optimization step (1 epoch) on a batch of data.

    Assert that:
    1. Weights after the update are strictly different from initial weights (assert not np.array_equal).
    2. Biases after the update are strictly different from initial biases.
    3. Parameters remain finite (no NaN or Inf values).
    """
    net = LinearNeuralNetwork([4, 6, 2])
    x_batch = np.random.randn(5, 4)
    y_batch = np.random.uniform(0.1, 0.9, size=(5, 2))

    w_before = [w.copy() for w in net.weights]
    b_before = [b.copy() for b in net.biases]

    # Run 1 epoch
    net.train(x_batch, y_batch, epochs=1, lr=0.1)

    for layer in range(net.num_layers):
        # Weights must be updated
        assert not np.array_equal(w_before[layer], net.weights[layer]), (
            f"Layer {layer} weights were not updated after train step"
        )

        # Biases must be updated
        assert not np.array_equal(b_before[layer], net.biases[layer]), (
            f"Layer {layer} biases were not updated after train step"
        )

        # Numerical sanity
        assert not np.isnan(net.weights[layer]).any(), f"NaN detected in layer {layer} weights"
        assert not np.isnan(net.biases[layer]).any(), f"NaN detected in layer {layer} biases"
        assert not np.isinf(net.weights[layer]).any(), f"Inf detected in layer {layer} weights"
        assert not np.isinf(net.biases[layer]).any(), f"Inf detected in layer {layer} biases"


def test_train_tiny_dataset_overfitting():
    """Sanity Check: Create a trivial dataset (2 samples) and train for 500 to 1000 epochs.

    Assert that the final loss drops below a strict threshold (< 1e-2).
    """

    np.random.seed(42)
    net = LinearNeuralNetwork([2, 4, 1])

    # 2 trivial samples
    x_train = np.array([[0.0, 1.0], [1.0, 0.0]])
    y_train = np.array([[0.1], [0.9]])

    # Record initial loss
    init_pred = net.forward(x_train)[-1]
    init_loss = math_helpers.mse(y_train, init_pred)

    # Train for 800 epochs
    net.train(x_train, y_train, epochs=800, lr=0.5)

    # Record final loss
    final_pred = net.forward(x_train)[-1]
    final_loss = math_helpers.mse(y_train, final_pred)

    assert final_loss < init_loss, (
        f"Loss failed to decrease: initial={init_loss:.4f}, final={final_loss:.4f}"
    )
    assert final_loss < 0.01, (
        f"Final loss ({final_loss:.6f}) did not drop below threshold (0.01)"
    )
