"""
Stage 1 Tests: Neural Network Initialization (init)
Tests weight/bias matrix shapes, statistical scaling distributions, and zero-bias initialization.
"""

import numpy as np
import pytest
from LinearNeuralNetwork import LinearNeuralNetwork


@pytest.mark.parametrize(
    "layer_sizes",
    [
        [4, 3],                  # Single-layer transition (input -> output)
        [4, 8, 3],               # Multi-layer perceptron (input -> hidden -> output)
        [784, 128, 64, 10],      # Standard MNIST classifier architecture
        [10, 5, 2, 1],           # Deep regression architecture
    ],
)
def test_init_shape_assertion(layer_sizes):
    """Pass specific layer dimensions and assert:

    - Weight shape is (layer_sizes[l], layer_sizes[l+1])
    - Bias shape is (layer_sizes[l+1],) or (1, layer_sizes[l+1])
    - Total number of layers matches len(layer_sizes) - 1
    """
    net = LinearNeuralNetwork(layer_sizes)

    expected_num_layers = len(layer_sizes) - 1
    assert len(net.weights) == expected_num_layers, (
        f"Expected {expected_num_layers} weight matrices, found {len(net.weights)}"
    )
    assert len(net.biases) == expected_num_layers, (
        f"Expected {expected_num_layers} bias vectors, found {len(net.biases)}"
    )

    for i in range(expected_num_layers):
        in_dim = layer_sizes[i]
        out_dim = layer_sizes[i + 1]

        # Weight shape assertion
        assert net.weights[i].shape == (in_dim, out_dim), (
            f"Layer {i}: expected weight shape {(in_dim, out_dim)}, got {net.weights[i].shape}"
        )

        # Bias shape assertion
        assert net.biases[i].shape in [(out_dim,), (1, out_dim)], (
            f"Layer {i}: expected bias shape {(out_dim,)} or {(1, out_dim)}, got {net.biases[i].shape}"
        )


def test_init_scale_and_distribution():
    """Verify statistical scale and distribution of initialized weights:

    - Mean should be approximately 0.0
    - Standard deviation should match the scaling formula: sqrt(2 / input_dim) (He normal)
    """
    np.random.seed(42)
    in_dim = 1000
    out_dim = 1000
    net = LinearNeuralNetwork([in_dim, out_dim])

    w = net.weights[0]
    actual_mean = float(np.mean(w))
    actual_std = float(np.std(w))
    expected_std = np.sqrt(2.0 / in_dim)  # He initialization formula used in LinearNeuralNetwork

    # Mean check (law of large numbers across 1,000,000 weights)
    assert np.isclose(actual_mean, 0.0, atol=0.01), (
        f"Expected weight mean ~ 0.0, got {actual_mean:.5f}"
    )

    # Scale check
    assert np.isclose(actual_std, expected_std, rtol=0.05), (
        f"Expected weight std ~ {expected_std:.5f}, got {actual_std:.5f}"
    )


@pytest.mark.parametrize(
    "layer_sizes",
    [
        [4, 3],
        [4, 8, 3],
        [16, 32, 16, 2],
    ],
)
def test_init_bias_zero_check(layer_sizes):
    """Assert that biases are initialized entirely to zeros unless specified otherwise."""
    net = LinearNeuralNetwork(layer_sizes)

    for idx, bias in enumerate(net.biases):
        assert np.all(bias == 0.0), (
            f"Layer {idx} bias contains non-zero elements: {bias}"
        )
