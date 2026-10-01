# 🧠 28×28 Neural Network Canvas Visualizer

An interactive Streamlit visualizer for teaching Neural Networks from scratch (e.g., MNIST digit classification).

## 🚀 Quickstart

Run the Streamlit application using `uv`:

```bash
uv run streamlit run app.py
```

or via:

```bash
uv run python main.py
```

---

## 🎨 How It Works

1. **Drawing Surface (280×280 px):**
   Students can draw a digit (0–9) using a white stroke on a black canvas with adjustable brush sizes.
2. **Visible 28×28 Downscaled View:**
   Shows the real-time downscaled $28 \times 28$ grayscale representation (normalized to $[0.0, 1.0]$) that the neural network actually processes.
3. **Numerical Matrix Inspector:**
   Inspect the raw pixel intensity values to reinforce that images are 2D arrays of numbers.
4. **Live Neural Network Feed:**
   Feeds the image into `LinearNeuralNetwork.py`. As soon as students implement `forward(x)`, predictions and probability distributions render in real time.

---

## 📝 For Students: What You Need to Implement

All your work is inside [`LinearNeuralNetwork.py`](LinearNeuralNetwork.py). You only need to fill in these 4 methods:

```python
class LinearNeuralNetwork:
    def __init__(self):
        # 1. Initialize weights (W) and biases (b)
        pass

    def forward(self, x):
        # 2. Compute z = Wx + b and activations (e.g., Sigmoid / ReLU / Softmax)
        ...

    def backward(self):
        # 3. Compute gradients via backpropagation
        ...

    def train(self):
        # 4. Update parameters using gradient descent
        ...
```

The Streamlit app hot-reloads dynamically when you save `LinearNeuralNetwork.py`!
