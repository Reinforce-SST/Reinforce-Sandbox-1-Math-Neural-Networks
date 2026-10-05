"""
Stage 2 Tests: Forward Propagation (forward)
Tests input/output matrix shapes, activation function bounds, and hardcoded mathematical calculations.
"""

import numpy as np
import pytest
from LinearNeuralNetwork import LinearNeuralNetwork
import math_helpers


# =====================================================================
# Activation Helper Functions for Validation
# =====================================================================

def relu(x: np.ndarray) -> np.ndarray:
    """Standard ReLU activation: max(0, x)."""
    return np.maximum(0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable Softmax activation: exp(x - max(x)) / sum(exp)."""
    shift_x = x - np.max(x, axis=-1, keepdims=True)
    exps = np.exp(shift_x)
    return exps / np.sum(exps, axis=-1, keepdims=True)


# =====================================================================
# Forward Propagation Tests
# =====================================================================

@pytest.mark.parametrize("batch_size", [1, 8, 32])
@pytest.mark.parametrize(
    "layer_sizes",
    [
        [4, 3],
        [4, 8, 3],
        [10, 20, 15, 2],
    ],
)
def test_forward_output_shape(batch_size, layer_sizes):
    """Feed an input batch of shape (batch_size, input_dim) through the network.

    Assert that:
    1. The final layer output shape matches (batch_size, output_dim).
    2. All intermediate post_activations match (batch_size, layer_sizes[i]).
    """
    net = LinearNeuralNetwork(layer_sizes)
    input_dim = layer_sizes[0]
    output_dim = layer_sizes[-1]

    x_input = np.random.randn(batch_size, input_dim)
    post_activations = net.forward(x_input)

    # post_activations contains [input, a_layer0, a_layer1, ..., a_layerL]
    assert len(post_activations) == len(layer_sizes), (
        f"Expected {len(layer_sizes)} post_activations elements, got {len(post_activations)}"
    )

    # Output shape assertion
    y_pred = post_activations[-1]
    assert y_pred.shape == (batch_size, output_dim), (
        f"Expected output shape {(batch_size, output_dim)}, got {y_pred.shape}"
    )

    # Intermediate layer activation shapes
    for i, act in enumerate(post_activations):
        expected_shape = (batch_size, layer_sizes[i])
        assert act.shape == expected_shape, (
            f"Activation at index {i} has shape {act.shape}, expected {expected_shape}"
        )


def test_forward_activation_ranges():
    """Pass random inputs across wide ranges through activation functions and assert constraints:

    - Sigmoid: output strictly between 0 and 1; sigmoid(0) == 0.5.
    - ReLU: output >= 0; negative inputs zeroed; positive inputs untouched.
    - Softmax: each output in [0, 1]; rows sum strictly to 1.0.
    - Network output: multi-layer network activations with sigmoid are within [0, 1].
    """
    np.random.seed(42)
    inputs = np.random.uniform(-50.0, 50.0, size=(100, 10))

    # 1. Sigmoid verification (math_helpers.py)
    # General range: across any inputs, float64 outputs must be in [0.0, 1.0]
    sig_out = math_helpers.sigmoid(inputs)
    assert np.all(sig_out >= 0.0) and np.all(sig_out <= 1.0), (
        "Sigmoid values must be bounded within [0, 1]"
    )
    assert np.isclose(math_helpers.sigmoid(0.0), 0.5), "Sigmoid(0) must equal 0.5"

    # Strict bounds: for moderate inputs (e.g. in [-10, 10]), float64 does not saturate
    moderate_inputs = np.random.uniform(-10.0, 10.0, size=(100, 10))
    sig_mod = math_helpers.sigmoid(moderate_inputs)
    assert np.all(sig_mod > 0.0) and np.all(sig_mod < 1.0), (
        "Sigmoid values must be strictly in (0, 1) for non-saturating inputs"
    )

    # 2. ReLU verification
    relu_out = relu(inputs)
    assert np.all(relu_out >= 0.0), "ReLU output must be >= 0"
    assert np.all(relu_out[inputs < 0] == 0.0), "Negative values must map to 0 in ReLU"
    assert np.allclose(relu_out[inputs > 0], inputs[inputs > 0]), "Positive values must remain unchanged in ReLU"

    # 3. Softmax verification
    softmax_out = softmax(inputs)
    assert np.all(softmax_out >= 0.0) and np.all(softmax_out <= 1.0), (
        "Softmax probabilities must be in range [0, 1]"
    )
    row_sums = np.sum(softmax_out, axis=1)
    assert np.allclose(row_sums, 1.0), "Softmax probabilities must sum to 1.0 along rows"

    # 4. Neural Network forward output range check
    net = LinearNeuralNetwork([5, 8, 3])
    out = net.forward(inputs[:, :5])[-1]
    assert np.all(out >= 0.0) and np.all(out <= 1.0), (
        "Network output using sigmoid activation must be bounded in [0, 1]"
    )


def test_forward_hardcoded_value():
    """Set weights and biases to fixed constants (W=1.0, b=0.0) and a fixed input vector.

    Assert that the forward pass matches exact manual paper-and-pencil calculations:

    Architecture: [2, 2, 1]
    Input: x = [[1.0, 1.0]]
    Layer 0 (Linear):
        z0 = [1.0, 1.0] @ [[1, 1], [1, 1]] + [0, 0] = [2.0, 2.0]
        a0 = [2.0, 2.0]  (LinearNeuralNetwork applies no activation for layer 0)
    Layer 1 (Sigmoid):
        z1 = [2.0, 2.0] @ [[1.0], [1.0]] + [0.0] = [4.0]
        a1 = sigmoid(4.0) = 1 / (1 + exp(-4)) ~ 0.98201379
    """
    net = LinearNeuralNetwork([2, 2, 1])

    # Fix Layer 0 parameters
    net.weights[0] = np.ones((2, 2), dtype=float)
    net.biases[0] = np.zeros(2, dtype=float)

    # Fix Layer 1 parameters
    net.weights[1] = np.ones((2, 1), dtype=float)
    net.biases[1] = np.zeros(1, dtype=float)

    x_input = np.array([[1.0, 1.0]])

    expected_layer0 = np.array([[2.0, 2.0]])
    expected_layer1 = np.array([[1.0 / (1.0 + np.exp(-4.0))]])

    post_activations = net.forward(x_input)

    # Layer 0 output check
    assert np.allclose(post_activations[1], expected_layer0), (
        f"Layer 0 output mismatch: expected {expected_layer0}, got {post_activations[1]}"
    )

    # Final Layer 1 output check
    assert np.allclose(post_activations[2], expected_layer1, atol=1e-6), (
        f"Final output mismatch: expected {expected_layer1}, got {post_activations[2]}"
    )
