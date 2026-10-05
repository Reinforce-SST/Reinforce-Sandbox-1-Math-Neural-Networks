import base64
import io
from pathlib import Path
import struct
import sys
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

# Import LinearNeuralNetwork and math helpers
from LinearNeuralNetwork import LinearNeuralNetwork
from math_helpers import mse

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
    "Interactive deep learning playground powered by `LinearNeuralNetwork.py`. "
    "Configure hidden layers, train on MNIST, and test predictions live on the drawing canvas."
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
# IDX Dataset Loader
# ---------------------------------------------------------
def load_mnist_images(filename):
    """Loads MNIST images from a ubyte file and normalizes them to [0, 1]."""
    with open(filename, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8)
        return data.reshape(num, rows * cols).astype(np.float32) / 255.0


def load_mnist_labels(filename):
    """Loads MNIST labels from a ubyte file."""
    with open(filename, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        buffer = f.read()
        return np.frombuffer(buffer, dtype=np.uint8)


@st.cache_data(show_spinner=False)
def load_dataset():
    """Loads MNIST train and test sets from dataset directory."""
    dataset_dir = Path(__file__).parent / "dataset"

    train_img_path = dataset_dir / "train-images.idx3-ubyte"
    if not train_img_path.exists():
        train_img_path = dataset_dir / "train-images-idx3-ubyte" / "train-images-idx3-ubyte"

    train_lbl_path = dataset_dir / "train-labels.idx1-ubyte"
    if not train_lbl_path.exists():
        train_lbl_path = dataset_dir / "train-labels-idx1-ubyte" / "train-labels-idx1-ubyte"

    test_img_path = dataset_dir / "t10k-images.idx3-ubyte"
    if not test_img_path.exists():
        test_img_path = dataset_dir / "t10k-images-idx3-ubyte" / "t10k-images-idx3-ubyte"

    test_lbl_path = dataset_dir / "t10k-labels.idx1-ubyte"
    if not test_lbl_path.exists():
        test_lbl_path = dataset_dir / "t10k-labels-idx1-ubyte" / "t10k-labels-idx1-ubyte"

    x_train = load_mnist_images(str(train_img_path))
    y_train_raw = load_mnist_labels(str(train_lbl_path))
    y_train = np.eye(10)[y_train_raw]

    x_test = load_mnist_images(str(test_img_path))
    y_test_raw = load_mnist_labels(str(test_lbl_path))
    y_test = np.eye(10)[y_test_raw]

    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------
# Sidebar: Model Architecture & Training Controls
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Model Architecture")

    # Fixed input and output dimensions for MNIST
    INPUT_SIZE = 784
    OUTPUT_SIZE = 10

    col_io1, col_io2 = st.columns(2)
    with col_io1:
        st.metric("Input Layer", f"{INPUT_SIZE}", help="Fixed: 28×28 pixels")
    with col_io2:
        st.metric("Output Layer", f"{OUTPUT_SIZE}", help="Fixed: 10 digit classes (0–9)")

    # Editable Hidden Layer Architecture
    hidden_mode = st.radio(
        "Hidden Layer Setup",
        options=["Preset Architectures", "Custom Hidden Sizes"],
        index=0,
    )

    if hidden_mode == "Preset Architectures":
        arch_choice = st.selectbox(
            "Select Preset",
            options=[
                "128 -> 32 (Default)",
                "128",
                "256 -> 128",
                "64 -> 32 -> 16",
                "64",
            ],
            index=0,
        )
        if arch_choice == "128 -> 32 (Default)":
            hidden_layers = [128, 32]
        elif arch_choice == "128":
            hidden_layers = [128]
        elif arch_choice == "256 -> 128":
            hidden_layers = [256, 128]
        elif arch_choice == "64 -> 32 -> 16":
            hidden_layers = [64, 32, 16]
        elif arch_choice == "64":
            hidden_layers = [64]
        else:
            hidden_layers = [128, 32]
    else:
        custom_str = st.text_input(
            "Custom Hidden Layers",
            value="128, 32",
            help="Comma-separated integers for hidden layer sizes, e.g., '128, 32' or '64' or '256, 128, 64'",
        )
        try:
            hidden_layers = [int(s.strip()) for s in custom_str.split(",") if s.strip()]
            if not hidden_layers or any(x <= 0 for x in hidden_layers):
                st.error("Invalid layer configuration. Falling back to [128, 32].")
                hidden_layers = [128, 32]
        except ValueError:
            st.error("Please enter valid comma-separated integers. Falling back to [128, 32].")
            hidden_layers = [128, 32]

    # Full network layer sizes: [Input, ...Hidden, Output]
    current_layer_sizes = [INPUT_SIZE] + hidden_layers + [OUTPUT_SIZE]
    arch_display = " ➔ ".join([str(sz) for sz in current_layer_sizes])
    st.info(f"📐 **Architecture:** `{arch_display}`")

    st.divider()
    st.header("🏋️ Hyperparameters")

    epochs = int(st.number_input("Epochs", min_value=1, max_value=5000, value=20, step=1))
    lr = st.number_input("Learning Rate", min_value=0.001, max_value=10.0, value=1.0, step=0.1, format="%.3f")

    data_limit = st.select_slider(
        "Training Samples (Speed vs Accuracy)",
        options=[500, 1000, 2000, 5000, 10000],
        value=1000,
        help="Use a smaller subset for instant training during lectures or more samples for higher accuracy.",
    )

    col_btn1, col_btn2 = st.columns(2)
    start_train = col_btn1.button("▶️ Start Training", type="primary", use_container_width=True)
    reset_btn = col_btn2.button("🔄 Reset Model", use_container_width=True)

    st.divider()
    st.header("🎨 Canvas & Preprocessing")
    stroke_width = st.slider("Brush Width", min_value=12, max_value=36, value=22, step=2)
    preprocess_mode = st.radio(
        "Preprocessing Method",
        options=["MNIST Center-of-Mass (Recommended)", "Raw Rescale (Direct 28x28)"],
        index=0,
        help="MNIST Center-of-Mass crops, scales to 20x20, and centers by center-of-mass, exactly matching MNIST training data.",
    )
    selected_mode = "mnist_center" if "Center-of-Mass" in preprocess_mode else "raw_rescale"

    st.divider()
    # Model status display
    if "is_trained" in st.session_state and st.session_state.is_trained:
        st.success("🟢 **Model Status:** Trained")
    else:
        st.info("🟡 **Model Status:** Untrained (Random Weights)")


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if (
    "model" not in st.session_state
    or st.session_state.get("current_layer_sizes") != current_layer_sizes
):
    st.session_state.model = LinearNeuralNetwork(current_layer_sizes)
    st.session_state.current_layer_sizes = current_layer_sizes
    st.session_state.is_trained = False
    st.session_state.training_history = []
    st.session_state.test_metrics = None
    st.session_state.last_training_summary = None

if reset_btn:
    st.session_state.model = LinearNeuralNetwork(current_layer_sizes)
    st.session_state.current_layer_sizes = current_layer_sizes
    st.session_state.is_trained = False
    st.session_state.training_history = []
    st.session_state.test_metrics = None
    st.session_state.last_training_summary = None
    st.success("Model reset with newly initialized weights.")
    st.rerun()


# ---------------------------------------------------------
# Training Dashboard Section (Persistent On Screen)
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

        model = LinearNeuralNetwork(current_layer_sizes)

        train_container = st.container()
        with train_container:
            st.subheader("🏋️ Training in Progress...")
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.markdown("**📉 Loss Curve (Training)**")
                loss_placeholder = st.empty()
            with col_chart2:
                st.markdown("**📈 Accuracy Curve (%)**")
                acc_placeholder = st.empty()

        history = []

        for epoch in range(epochs):
            # Pure full-batch Gradient Descent
            posts = model.forward(x_train_sub)
            model.backward(y_train_sub, posts, lr=lr)

            # Evaluate training metrics after epoch
            train_preds = posts[-1]
            loss_val = float(mse(y_train_sub, train_preds))
            acc_val = float(np.mean(np.argmax(train_preds, axis=1) == np.argmax(y_train_sub, axis=1)))

            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.markdown(
                f"**Epoch `{epoch + 1}/{epochs}`** &nbsp;|&nbsp; "
                f"**Loss:** `{loss_val:.4f}` &nbsp;|&nbsp; "
                f"**Train Accuracy:** `{acc_val * 100:.2f}%`"
            )
            history.append({
                "Epoch": epoch + 1,
                "Loss": loss_val,
                "Accuracy (%)": acc_val * 100,
            })
            chart_df = pd.DataFrame(history).set_index("Epoch")
            loss_placeholder.line_chart(chart_df["Loss"])
            acc_placeholder.line_chart(chart_df["Accuracy (%)"])

        # Evaluate on test set (or subset for speed)
        test_sub_size = min(2000, len(x_test))
        test_posts = model.forward(x_test[:test_sub_size])
        test_preds = test_posts[-1]
        test_loss = float(mse(y_test[:test_sub_size], test_preds))
        test_acc = float(np.mean(np.argmax(test_preds, axis=1) == np.argmax(y_test[:test_sub_size], axis=1)))

        summary_msg = f"🎉 Training Complete! Tested on {test_sub_size} samples ➔ Accuracy: {test_acc * 100:.2f}% | Loss: {test_loss:.4f}"

        st.session_state.model = model
        st.session_state.is_trained = True
        st.session_state.training_history = history
        st.session_state.test_metrics = {"loss": test_loss, "accuracy": test_acc}
        st.session_state.last_training_summary = summary_msg

        st.success(summary_msg)
    except Exception as e:
        st.error(f"Error during training: {e}")

elif st.session_state.training_history:
    # Persistently display training results & curves on screen after training is done
    st.subheader("🏋️ Training Performance & History")
    if st.session_state.last_training_summary:
        st.success(st.session_state.last_training_summary)

    hist_df = pd.DataFrame(st.session_state.training_history).set_index("Epoch")
    final_loss = float(hist_df["Loss"].iloc[-1])
    final_acc = float(hist_df["Accuracy (%)"].iloc[-1])
    tm = st.session_state.test_metrics

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Final Train Loss", f"{final_loss:.4f}")
    col_m2.metric("Final Train Accuracy", f"{final_acc:.2f}%")
    if tm:
        col_m3.metric("Test Accuracy", f"{tm['accuracy']*100:.2f}%")
        col_m4.metric("Test Loss", f"{tm['loss']:.4f}")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("**📉 Loss Curve (Training)**")
        st.line_chart(hist_df["Loss"], height=230)
    with col_c2:
        st.markdown("**📈 Accuracy Curve (%)**")
        st.line_chart(hist_df["Accuracy (%)"], height=230)

    st.divider()


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
    st.caption("Live feed through `LinearNeuralNetwork` forward pass")

    x_input = img_28.reshape(1, 784)
    active_model = st.session_state.model

    if not st.session_state.is_trained:
        st.warning("⚠️ **Model is currently untrained.** Click **'▶️ Start Training'** in the sidebar to train on MNIST.")

    try:
        # Run forward inference through LinearNeuralNetwork
        post_activations = active_model.forward(x_input)
        raw_output = post_activations[-1]
        flat_out = np.squeeze(raw_output)

        # Calibrated Probability Calculation:
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
        num_layers = active_model.num_layers
        st.markdown(f"**Total Weight Layers:** `{num_layers}`")
        for i in range(num_layers):
            w = active_model.weights[i]
            b = active_model.biases[i]
            in_dim = w.shape[0]
            out_dim = w.shape[1]
            st.markdown(
                f"- **Layer {i+1}:** `({in_dim} → {out_dim})` | "
                f"Weights: `{w.shape}` | Biases: `{b.shape}`"
            )

with col_right:
    with st.expander("⚡ Live Layer Activations for Canvas Input", expanded=False):
        if has_drawing:
            try:
                live_posts = active_model.forward(x_input)
                for i, post_act in enumerate(live_posts):
                    if i == 0:
                        st.write(f"- **Input $a^{(0)}$:** shape `{post_act.shape}`, mean `{np.mean(post_act):.4f}`")
                    else:
                        st.write(
                            f"- **Layer {i} Activation $a^{{({i})}}$:** shape `{post_act.shape}`, "
                            f"min `{np.min(post_act):.3f}`, max `{np.max(post_act):.3f}`, mean `{np.mean(post_act):.3f}`"
                        )
            except Exception as e:
                st.write(f"Error computing activations: {e}")
        else:
            st.write("Draw on canvas to trigger activation computation.")


