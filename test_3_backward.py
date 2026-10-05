"""
Stage 3 Tests: Backpropagation (backward)
Tests gradient matrix dimensions, parameter delta shapes, and gradient flow on non-zero error.
"""

import numpy as np
import pytest
from LinearNeuralNetwork import LinearNeuralNetwork


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize(
    "layer_sizes",
    [
        [4, 3],                  # 1 weight matrix
        [4, 8, 3],               # 2 weight matrices
        [10, 20, 15, 2],         # 3 weight matrices
    ],
)
def test_backward_gradient_shapes(batch_size, layer_sizes):
    """Verify that the calculated gradient matrices (grad_W, grad_b) have the exact

    same shapes as their corresponding parameter matrices (W, b).

    In LinearNeuralNetwork, backward performs in-place SGD parameter updates:
        W_after = W_before - lr * grad_W  =>  grad_W = (W_before - W_after) / lr
        b_after = b_before - lr * grad_b  =>  grad_b = (b_before - b_after) / lr
    """
    net = LinearNeuralNetwork(layer_sizes)
    in_dim = layer_sizes[0]
    out_dim = layer_sizes[-1]

    x_batch = np.random.randn(batch_size, in_dim)
    y_batch = np.random.uniform(0.1, 0.9, size=(batch_size, out_dim))

    post_activations = net.forward(x_batch)

    # Cache pre-backward parameter states
    w_before = [w.copy() for w in net.weights]
    b_before = [b.copy() for b in net.biases]

    lr = 0.25
    net.backward(y_batch, post_activations, lr=lr)

    # Inspect gradients derived from parameter updates
    for layer in range(net.num_layers):
        grad_w = (w_before[layer] - net.weights[layer]) / lr
        grad_b = (b_before[layer] - net.biases[layer]) / lr

        # 1. Assert gradient shapes match parameter shapes exactly
        assert grad_w.shape == w_before[layer].shape, (
            f"Layer {layer}: grad_W shape {grad_w.shape} != weight shape {w_before[layer].shape}"
        )
        assert grad_b.shape == b_before[layer].shape, (
            f"Layer {layer}: grad_b shape {grad_b.shape} != bias shape {b_before[layer].shape}"
        )

        # 2. Assert gradients are finite numbers (no NaN or Inf)
        assert np.all(np.isfinite(grad_w)), f"Layer {layer} grad_W contains NaN or Inf"
        assert np.all(np.isfinite(grad_b)), f"Layer {layer} grad_b contains NaN or Inf"


def test_backward_nonzero_gradients_on_prediction_error():
    """Assert that backward produces non-zero gradients across all layers when predictions differ from targets."""
    net = LinearNeuralNetwork([3, 5, 2])
    x = np.ones((4, 3))
    y = np.zeros((4, 2))  # Distinct from sigmoid output (> 0.0), ensuring non-zero loss

    post_activations = net.forward(x)
    w_before = [w.copy() for w in net.weights]
    b_before = [b.copy() for b in net.biases]

    lr = 0.1
    net.backward(y, post_activations, lr=lr)

    for layer in range(net.num_layers):
        grad_w = (w_before[layer] - net.weights[layer]) / lr
        grad_b = (b_before[layer] - net.biases[layer]) / lr

        assert not np.allclose(grad_w, 0.0), f"Layer {layer} grad_W is all zeros"
        assert not np.allclose(grad_b, 0.0), f"Layer {layer} grad_b is all zeros"


def test_backward_batch_aggregation():
    """Ensure bias gradients properly aggregate across the batch dimension.

    For a batch of size N, dz_db is summed across axis=0, producing a shape (out_dim,)
    matching the layer bias shape, not (batch_size, out_dim).
    """
    batch_size = 10
    layer_sizes = [4, 6, 2]
    net = LinearNeuralNetwork(layer_sizes)

    x = np.random.randn(batch_size, 4)
    y = np.random.uniform(0.1, 0.9, size=(batch_size, 2))

    post_activations = net.forward(x)
    b_before = [b.copy() for b in net.biases]

    lr = 0.1
    net.backward(y, post_activations, lr=lr)

    for layer in range(net.num_layers):
        grad_b = (b_before[layer] - net.biases[layer]) / lr
        assert grad_b.ndim == 1, (
            f"Layer {layer} bias gradient should be 1-dimensional (aggregated across batch), got {grad_b.ndim}D"
        )
        assert grad_b.shape[0] == layer_sizes[layer + 1], (
            f"Layer {layer} bias gradient dimension mismatch: {grad_b.shape[0]} != {layer_sizes[layer + 1]}"
        )
