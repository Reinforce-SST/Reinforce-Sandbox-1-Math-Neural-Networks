import base64
import io
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

# Ensure neural_network module and its internal dependencies are discoverable
NEURAL_NET_DIR = Path(__file__).parent / "neural_network"
if str(NEURAL_NET_DIR) not in sys.path:
    sys.path.insert(0, str(NEURAL_NET_DIR))

from activation import Activation
from loss_function import LossFunction
from optimizer import Optimizer
from neural_network import NeuralNetwork
from loader import load_mnist_images, load_mnist_labels

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Neural Network Sandbox: Training & Inference",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Neural Network Sandbox: Live Training & Canvas Inference")
st.caption(
    "Interactive deep learning playground connected directly to `neural_network/`. "
    "Train your custom multi-layer perceptron on MNIST and test predictions live on the drawing canvas."
)

# Declare HTML5 Canvas Component
CANVAS_DIR = Path(__file__).parent / "canvas_component"
digit_canvas = components.declare_component("digit_canvas", path=str(CANVAS_DIR))


# ---------------------------------------------------------
# MNIST Preprocessing & Centering (Center of Mass & Bounding Box)
# ---------------------------------------------------------
def preprocess_mnist_digit(pil_img, mode="mnist_center"):
    """
    Preprocesses canvas drawing to match standard MNIST distribution:
    1. Identifies the bounding box of drawn pixels.
    2. Scales the digit into a 20x20 pixel box (preserving aspect ratio).
    3. Centers the digit in a 28x28 canvas using Center of Mass (same as MNIST).
    """
    arr = np.array(pil_img, dtype=np.float32)
    if not np.any(arr > 20):
        return np.zeros((28, 28), dtype=np.float32), False

    if mode == "raw_rescale":
        pil_28 = pil_img.resize((28, 28), Image.Resampling.BILINEAR)
        img_28 = np.array(pil_28, dtype=np.float32) / 255.0
        return img_28, True

    # 1. Bounding box cropping
    rows = np.any(arr > 20, axis=1)
    cols = np.any(arr > 20, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    cropped = arr[rmin : rmax + 1, cmin : cmax + 1]
    h, w = cropped.shape

    # 2. Aspect-ratio preserving resize into 20x20
    if h > w:
        new_h = 20
        new_w = max(1, int(round((w / h) * 20)))
    else:
        new_w = 20
        new_h = max(1, int(round((h / w) * 20)))

    cropped_pil = Image.fromarray(cropped.astype(np.uint8))
    resized_pil = cropped_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
    resized_arr = np.array(resized_pil, dtype=np.float32)

    # 3. Paste into 28x28 black image canvas
    padded = np.zeros((28, 28), dtype=np.float32)
    pad_top = (28 - new_h) // 2
    pad_left = (28 - new_w) // 2
    padded[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized_arr

    # 4. Center of Mass Shift
    total_mass = np.sum(padded)
    if total_mass > 0:
        cy, cx = np.meshgrid(np.arange(28), np.arange(28), indexing="ij")
        center_y = np.sum(cy * padded) / total_mass
        center_x = np.sum(cx * padded) / total_mass

        shift_y = int(round(13.5 - center_y))
        shift_x = int(round(13.5 - center_x))

        # Apply integer translation
        shifted = np.zeros_like(padded)
        for y in range(28):
            for x in range(28):
                ny, nx = y + shift_y, x + shift_x
                if 0 <= ny < 28 and 0 <= nx < 28:
                    shifted[ny, nx] = padded[y, x]
        padded = shifted

    img_28 = np.clip(padded / 255.0, 0.0, 1.0)
    return img_28, True


# ---------------------------------------------------------
# Cached Dataset Loader
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_dataset():
    """Loads MNIST train and test sets from neural_network/dataset."""
    dataset_dir = NEURAL_NET_DIR / "dataset"

    # Path resolution supporting both directory nested or flat structures
    train_img_path = dataset_dir / "train-images-idx3-ubyte" / "train-images-idx3-ubyte"
    if not train_img_path.exists():
        train_img_path = dataset_dir / "train-images.idx3-ubyte"

    train_lbl_path = dataset_dir / "train-labels-idx1-ubyte" / "train-labels-idx1-ubyte"
    if not train_lbl_path.exists():
        train_lbl_path = dataset_dir / "train-labels.idx1-ubyte"

    test_img_path = dataset_dir / "t10k-images-idx3-ubyte" / "t10k-images-idx3-ubyte"
    if not test_img_path.exists():
        test_img_path = dataset_dir / "t10k-images.idx3-ubyte"

    test_lbl_path = dataset_dir / "t10k-labels-idx1-ubyte" / "t10k-labels-idx1-ubyte"
    if not test_lbl_path.exists():
        test_lbl_path = dataset_dir / "t10k-labels.idx1-ubyte"

    x_train = load_mnist_images(str(train_img_path))
    y_train_raw = load_mnist_labels(str(train_lbl_path))
    y_train = np.eye(10)[y_train_raw]

    x_test = load_mnist_images(str(test_img_path))
    y_test_raw = load_mnist_labels(str(test_lbl_path))
    y_test = np.eye(10)[y_test_raw]

    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
def create_default_network(hidden_layers_spec=None):
    if hidden_layers_spec is None:
        hidden_layers_spec = [
            (128, Activation.SIGMOID),
            (32, Activation.SIGMOID),
        ]
    return NeuralNetwork(input_size=784, output_size=10, hidden_layers=hidden_layers_spec)


if "model" not in st.session_state:
    st.session_state.model = create_default_network()
    st.session_state.is_trained = False
    st.session_state.training_history = []
    st.session_state.test_metrics = None


# ---------------------------------------------------------
# Sidebar: Training Controls & Canvas Settings
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Model Training")

    arch_choice = st.selectbox(
        "Hidden Layers Architecture",
        options=[
            "128 -> 32 (Default)",
            "128",
            "256 -> 128",
            "64 -> 32 -> 16",
        ],
        index=0,
    )

    act_choice = st.selectbox(
        "Activation Function",
        options=["SIGMOID", "RELU", "TANH", "LINEAR"],
        index=0,
    )
    act_enum = Activation[act_choice]

    if arch_choice == "128 -> 32 (Default)":
        hidden_config = [(128, act_enum), (32, act_enum)]
    elif arch_choice == "128":
        hidden_config = [(128, act_enum)]
    elif arch_choice == "256 -> 128":
        hidden_config = [(256, act_enum), (128, act_enum)]
    elif arch_choice == "64 -> 32 -> 16":
        hidden_config = [(64, act_enum), (32, act_enum), (16, act_enum)]
    else:
        hidden_config = [(128, act_enum), (32, act_enum)]

    epochs = st.slider("Epochs", min_value=1, max_value=50, value=12, step=1)
    lr = st.number_input("Learning Rate", min_value=0.001, max_value=5.0, value=0.2, step=0.05, format="%.3f")
    batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128, 256], value=32)

    opt_choice = st.selectbox("Optimizer", options=["MINI_BATCH", "SGD", "GD"], index=0)
    optimizer_enum = Optimizer[opt_choice]

    loss_choice = st.selectbox("Loss Function", options=["LOG_LOSS", "MSE"], index=0)
    loss_enum = LossFunction[loss_choice]

    data_limit = st.select_slider(
        "Training Samples (Speed vs Accuracy)",
        options=[5000, 10000, 20000, 60000],
        value=20000,
        help="Use a smaller subset for faster training or 60,000 for full dataset accuracy."
    )

    col_btn1, col_btn2 = st.columns(2)
    start_train = col_btn1.button("▶️ Start Training", type="primary", use_container_width=True)
    reset_btn = col_btn2.button("🔄 Reset Model", use_container_width=True)

    if reset_btn:
        st.session_state.model = create_default_network(hidden_config)
        st.session_state.is_trained = False
        st.session_state.training_history = []
        st.session_state.test_metrics = None
        st.success("Model reset to initial random weights.")
        st.rerun()

    st.divider()
    st.header("🎨 Canvas & Preprocessing")
    stroke_width = st.slider("Brush Width", min_value=12, max_value=36, value=22, step=2)
    preprocess_mode = st.radio(
        "Preprocessing Method",
        options=["MNIST Center-of-Mass (Recommended)", "Raw Rescale (Direct 28x28)"],
        index=0,
        help="MNIST Center-of-Mass crops, scales to 20x20, and centers by center-of-mass, exactly matching MNIST training data."
    )
    selected_mode = "mnist_center" if "Center-of-Mass" in preprocess_mode else "raw_rescale"

    st.divider()
    if st.session_state.is_trained:
        st.success("🟢 **Model Status:** Trained")
    else:
        st.info("🟡 **Model Status:** Untrained (Random Weights)")


# ---------------------------------------------------------
# Training Execution Logic
# ---------------------------------------------------------
if start_train:
    st.info("⏳ Loading MNIST dataset...")
    try:
        x_train, y_train, x_test, y_test = load_dataset()

        if data_limit < len(x_train):
            indices = np.random.choice(len(x_train), data_limit, replace=False)
            x_train_sub = x_train[indices]
            y_train_sub = y_train[indices]
        else:
            x_train_sub = x_train
            y_train_sub = y_train

        model = NeuralNetwork(input_size=784, output_size=10, hidden_layers=hidden_config)

        train_container = st.container()
        with train_container:
            st.subheader("🏋️ Training in Progress...")
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            chart_placeholder = st.empty()

        history = []

        def training_callback(epoch_idx, total_epochs, loss_val, acc_val):
            progress = (epoch_idx + 1) / total_epochs
            progress_bar.progress(progress)
            status_text.markdown(
                f"**Epoch `{epoch_idx + 1}/{total_epochs}`** &nbsp;|&nbsp; "
                f"**Loss:** `{loss_val:.4f}` &nbsp;|&nbsp; "
                f"**Train Accuracy:** `{acc_val * 100:.2f}%`"
            )
            history.append({
                "Epoch": epoch_idx + 1,
                "Loss": loss_val,
                "Accuracy (%)": acc_val * 100,
            })
            chart_df = pd.DataFrame(history).set_index("Epoch")
            chart_placeholder.line_chart(chart_df[["Accuracy (%)", "Loss"]])

        model.train(
            input=x_train_sub,
            target=y_train_sub,
            epochs=epochs,
            learning_rate=lr,
            loss_func=loss_enum,
            optimizer=optimizer_enum,
            batch_size=batch_size,
            log=False,
            callback=training_callback,
        )

        test_preds = model.forward(x_test)
        test_loss = float(np.mean(loss_enum.compute(y_test, test_preds)).item())
        test_acc = float(np.mean(np.argmax(test_preds, axis=1) == np.argmax(y_test, axis=1)))

        st.session_state.model = model
        st.session_state.is_trained = True
        st.session_state.training_history = history
        st.session_state.test_metrics = {"loss": test_loss, "accuracy": test_acc}

        st.success(f"🎉 Training Complete! Test Accuracy: **{test_acc * 100:.2f}%** | Test Loss: **{test_loss:.4f}**")
    except Exception as e:
        st.error(f"Error during training: {e}")


# ---------------------------------------------------------
# Main UI Layout: 3 Columns
# Col 1: Drawing Canvas (280x280)
# Col 2: Processed 28x28 View & Matrix Stats
# Col 3: Neural Network Prediction & Probabilities
# ---------------------------------------------------------
col1, col2, col3 = st.columns([1.1, 1.1, 1.3], gap="large")

with col1:
    st.subheader("1. ✍️ Draw Here (280×280)")
    st.caption("Draw a digit (0–9) using the white brush on the canvas:")

    canvas_data = digit_canvas(stroke_width=stroke_width, key="digit_canvas_component")
    st.markdown("💡 *Tip: Draw digits normally anywhere on the canvas — auto-centering aligns them to MNIST standard.*")

# Extract and process image from canvas
has_drawing = False
img_28 = np.zeros((28, 28), dtype=np.float32)

if canvas_data and isinstance(canvas_data, str) and canvas_data.startswith("data:image"):
    try:
        header, encoded = canvas_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("L")

        # Process with bounding box + center of mass centering
        img_28, has_drawing = preprocess_mnist_digit(pil_img, mode=selected_mode)
    except Exception as e:
        st.error(f"Image processing error: {e}")

with col2:
    st.subheader("2. 🔬 28×28 Processed View")
    st.caption("Centered & scaled grayscale input fed into the neural network:")

    st.image(
        img_28,
        caption="Normalized [0.0, 1.0] 28×28 Input" if has_drawing else "Canvas empty (All zeros)",
        width=280,
        clamp=True,
    )

    nonzero_pixels = int(np.count_nonzero(img_28 > 0.05))
    st.markdown(
        f"**Shape:** `(28, 28)` &nbsp;|&nbsp; "
        f"**Non-zero pixels:** `{nonzero_pixels}/784` &nbsp;|&nbsp; "
        f"**Max:** `{img_28.max():.2f}`"
    )

    with st.expander("🔍 Inspect Raw 28×28 Numerical Matrix"):
        st.write("First 10×10 slice of pixel values:")
        st.dataframe(
            pd.DataFrame(np.round(img_28[:10, :10], 2)),
            use_container_width=True,
        )


with col3:
    st.subheader("3. 🚀 Neural Network Prediction")
    st.caption("Live feed through `neural_network/` forward pass")

    x_input = img_28.reshape(1, 784)
    active_model = st.session_state.model

    if not st.session_state.is_trained:
        st.warning("⚠️ **Model is currently untrained.** Click **'▶️ Start Training'** in the sidebar to train on MNIST.")

    try:
        # Run forward inference
        raw_output = active_model.forward(x_input)
        flat_out = np.squeeze(raw_output)

        # Calibrated Probability Calculation:
        # Since the network output is sigmoid (in [0, 1]), direct normalization gives true confidence
        sum_activations = float(np.sum(flat_out))
        if sum_activations > 1e-6 and np.all(flat_out >= 0):
            probs = flat_out / sum_activations
        else:
            exp_vals = np.exp(flat_out - np.max(flat_out))
            probs = exp_vals / np.sum(exp_vals)

        predicted_digit = int(np.argmax(flat_out))
        confidence = float(probs[predicted_digit]) * 100

        # Primary Metric Card
        st.metric(
            label="Predicted Digit",
            value=f"{predicted_digit}" if has_drawing else "—",
            delta=f"{confidence:.1f}% confidence" if has_drawing else "Draw a digit",
        )

        # Class Probability Bar Chart
        prob_df = pd.DataFrame({
            "Digit": [str(i) for i in range(10)],
            "Probability": probs if has_drawing else np.zeros(10),
            "Raw Activation": flat_out if has_drawing else np.zeros(10),
        })
        st.bar_chart(prob_df.set_index("Digit")["Probability"], height=220)

        if has_drawing:
            # Top 3 Candidates
            top3_indices = np.argsort(probs)[::-1][:3]
            top3_str = " &nbsp;|&nbsp; ".join(
                [f"**#{rank+1}:** `{idx}` ({probs[idx]*100:.1f}%)" for rank, idx in enumerate(top3_indices)]
            )
            st.markdown(f"**Top Candidates:** {top3_str}")
        else:
            st.info("Draw a digit on the left canvas to see predictions in real time.")

    except Exception as e:
        st.error(f"Inference error: {e}")


# ---------------------------------------------------------
# Educational Inspection Section: Layer Weights & Activations
# ---------------------------------------------------------
st.divider()
col_left, col_right = st.columns(2)

with col_left:
    with st.expander("📚 Model Architecture & Layer Dimensions", expanded=False):
        st.markdown(f"**Total Layers:** `{len(active_model.layers)}`")
        for i, layer in enumerate(active_model.layers):
            w = active_model.weights[i]
            b = active_model.biases[i]
            act_name = layer[2].name
            st.markdown(
                f"- **Layer {i+1}:** `({layer[0]} → {layer[1]})` | Activation: `{act_name}` | "
                f"Weights: `{w.shape}` | Biases: `{b.shape}`"
            )

with col_right:
    with st.expander("⚡ Live Layer Activations for Canvas Input", expanded=False):
        if hasattr(active_model, "posts") and len(active_model.posts) > 0 and has_drawing:
            for i, post_act in enumerate(active_model.posts):
                if i == 0:
                    st.write(f"- **Input $a^{(0)}$:** shape `{post_act.shape}`, mean `{np.mean(post_act):.4f}`")
                else:
                    st.write(
                        f"- **Layer {i} Activation $a^{{({i})}}$:** shape `{post_act.shape}`, "
                        f"min `{np.min(post_act):.3f}`, max `{np.max(post_act):.3f}`, mean `{np.mean(post_act):.3f}`"
                    )
        else:
            st.write("Draw on canvas to trigger activation computation.")

# Training Performance & Test Metrics Expander
if st.session_state.training_history:
    with st.expander("📈 Training History & Evaluation Metrics", expanded=False):
        hist_df = pd.DataFrame(st.session_state.training_history)
        if st.session_state.test_metrics:
            tm = st.session_state.test_metrics
            st.markdown(
                f"**Final Test Accuracy:** `{tm['accuracy']*100:.2f}%` &nbsp;|&nbsp; "
                f"**Final Test Loss:** `{tm['loss']:.4f}`"
            )
        st.line_chart(hist_df.set_index("Epoch")[["Accuracy (%)", "Loss"]])
