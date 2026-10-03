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

## 🎨 Features

1. **Integrated Training Dashboard**:
   - Configure hyperparameters directly from the sidebar: Hidden layers architecture (`128 -> 32`, `256 -> 128`, etc.), Activation (`SIGMOID`, `RELU`, `TANH`), Learning Rate, Epochs, Batch Size, Optimizer (`MINI_BATCH`, `SGD`, `GD`), and Loss Function (`LOG_LOSS`, `MSE`).
   - Click **"▶️ Start Training"** to train epoch-by-epoch with real-time progress bars, live loss & accuracy curves, and final test evaluation on the MNIST dataset.
2. **Drawing Surface (280×280 px):**
   - Draw digits (0–9) on the interactive HTML5 canvas with adjustable brush sizes.
3. **28×28 Grayscale Downscaled View:**
   - Real-time downscaled $28 \times 28$ matrix (normalized to $[0.0, 1.0]$) that is fed directly into `NeuralNetwork.forward(x)`.
4. **Live Inference & Probability Breakdown:**
   - Real-time predicted digit, confidence percentage, top-3 candidates, and class probability distribution bar chart (0–9).
5. **Model & Layer Inspector:**
   - Inspect layer dimensions, weight/bias matrix shapes, and live hidden layer activation values for drawn digits.
