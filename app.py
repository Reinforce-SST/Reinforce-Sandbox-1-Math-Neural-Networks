import base64
import importlib
import inspect
import io
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(
    page_title="28x28 Neural Network Visualizer",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 28×28 Digit Canvas & Neural Network Visualizer")
st.caption(
    "Interactive teaching sandbox: Draw on the canvas, inspect the downscaled 28×28 "
    "pixel grid, and feed it directly into `LinearNeuralNetwork.py`."
)

# Declare HTML5 Canvas Component
CANVAS_DIR = Path(__file__).parent / "canvas_component"
digit_canvas = components.declare_component("digit_canvas", path=str(CANVAS_DIR))

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Canvas Settings")
    stroke_width = st.slider("Brush Width", min_value=12, max_value=32, value=20, step=2)
    
    st.divider()
    st.header("📐 Input Format to `forward(x)`")
    target_shape = st.selectbox(
        "Vector shape passed to `forward(x)`",
        options=["(1, 784) - Row vector", "(784, 1) - Column vector", "(28, 28) - 2D Matrix"],
        index=0,
    )
    
    st.divider()
    st.header("🔄 Model Hot-Reload")
    reload_requested = st.button("Reload LinearNeuralNetwork.py", use_container_width=True)
    st.info("Live-code in `LinearNeuralNetwork.py` during class and click above or draw to see updates.")


# ---------------------------------------------------------
# Model Loading & Hot-Reloading
# ---------------------------------------------------------
def get_model(force_reload=False):
    """Dynamically imports and reloads LinearNeuralNetwork."""
    try:
        import LinearNeuralNetwork as lnn_module
        if force_reload or "LinearNeuralNetwork" in sys.modules:
            lnn_module = importlib.reload(lnn_module)
        
        cls = getattr(lnn_module, "LinearNeuralNetwork", None)
        if cls is None:
            return None, "Class `LinearNeuralNetwork` not found in `LinearNeuralNetwork.py`."
        
        # Instantiate model defensively
        sig = inspect.signature(cls.__init__)
        params = [p for p in sig.parameters.values() if p.name != "self"]
        has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
        required_params = [
            p for p in params 
            if p.default == inspect.Parameter.empty and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        
        if len(required_params) == 0 or has_varargs:
            model = cls()
        elif len(required_params) == 3:
            model = cls(784, 128, 10)
        else:
            model = cls(*([1] * len(required_params)))
            
        return model, None
    except Exception as e:
        return None, f"Error instantiating `LinearNeuralNetwork`: {e}"


model, model_err = get_model(force_reload=reload_requested)


# ---------------------------------------------------------
# Main Layout: 3 Columns
# Col 1: Drawing Canvas (280x280)
# Col 2: Processed 28x28 View & Matrix Stats
# Col 3: Neural Network Output & Activation Breakdown
# ---------------------------------------------------------
col1, col2, col3 = st.columns([1.1, 1.1, 1.3], gap="large")

with col1:
    st.subheader("1. ✍️ Draw Here (280×280)")
    st.caption("Draw a digit (0–9) using the white brush on the black canvas:")
    
    # Render custom HTML5 canvas component
    canvas_data = digit_canvas(stroke_width=stroke_width, key="digit_canvas_component")
    st.markdown("💡 *Tip: Center the digit for best MNIST-like representation.*")

# Extract and process image
has_drawing = False
img_28 = np.zeros((28, 28), dtype=np.float32)

if canvas_data and isinstance(canvas_data, str) and canvas_data.startswith("data:image"):
    try:
        header, encoded = canvas_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("L")
        # Downscale to 28x28
        pil_28 = pil_img.resize((28, 28), Image.Resampling.BILINEAR)
        # Normalize to [0.0, 1.0]
        img_28 = np.array(pil_28, dtype=np.float32) / 255.0
        if np.any(img_28 > 0.05):
            has_drawing = True
    except Exception as e:
        st.error(f"Image processing error: {e}")

with col2:
    st.subheader("2. 🔬 28×28 Processed View")
    st.caption("How your Neural Network actually sees the input:")
    
    # Display the 28x28 downsampled image crisply
    st.image(
        img_28,
        caption="Downscaled 28×28 Input Image (Normalized [0.0, 1.0])" if has_drawing else "Canvas empty (All zeros)",
        width=280,
        clamp=True,
    )
    
    # Metadata badges
    nonzero_pixels = int(np.count_nonzero(img_28 > 0.05))
    st.markdown(
        f"**Shape:** `(28, 28)` &nbsp;|&nbsp; "
        f"**Non-zero pixels:** `{nonzero_pixels}/784` &nbsp;|&nbsp; "
        f"**Max:** `{img_28.max():.2f}`"
    )
    
    with st.expander("🔍 Inspect Raw 28×28 Numerical Matrix"):
        st.write("First 10×10 slice of pixel intensities:")
        st.dataframe(
            pd.DataFrame(np.round(img_28[:10, :10], 2)),
            use_container_width=True,
        )


with col3:
    st.subheader("3. 🚀 Neural Network Prediction")
    st.caption("Live feed into `LinearNeuralNetwork.forward(x)`")
    
    # Prepare input vector according to selected format
    if "(1, 784)" in target_shape:
        x_input = img_28.reshape(1, 784)
    elif "(784, 1)" in target_shape:
        x_input = img_28.reshape(784, 1)
    else:
        x_input = img_28.copy()
        
    st.code(f"x shape passed to model: {x_input.shape}", language="python")

    if model_err:
        st.error(model_err)
    elif model is None:
        st.warning("⚠️ Model is not available.")
    else:
        # Check forward method signature and implementation
        forward_func = getattr(model, "forward", None)
        if forward_func is None:
            st.warning("`LinearNeuralNetwork` does not have a `forward` method.")
        else:
            forward_sig = inspect.signature(forward_func)
            params = [p for p in forward_sig.parameters.values() if p.name != "self"]
            
            # Execute forward defensively
            result = None
            forward_executed = False
            forward_err = None
            
            try:
                if len(params) == 0:
                    result = forward_func()
                    forward_executed = True
                else:
                    try:
                        result = forward_func(x_input)
                        forward_executed = True
                    except Exception as e_pass:
                        if x_input.ndim == 2:
                            result = forward_func(x_input.T)
                            forward_executed = True
                        else:
                            raise e_pass
            except Exception as e:
                forward_err = str(e)
            
            # Check if forward was not yet implemented (returns None or Ellipsis)
            if forward_executed and (result is None or result is Ellipsis):
                st.info(
                    "🎓 **Ready for Live Teaching!**\n\n"
                    "Implement the 4 functions in `LinearNeuralNetwork.py`:\n"
                    "- `__init__(self, ...)`: Initialize weights ($W$) and biases ($b$)\n"
                    "- `forward(self, x)`: Compute activations $z = Wx + b$, $a = \\sigma(z)$\n"
                    "- `backward(self, ...)`: Compute gradients\n"
                    "- `train(self, ...)`: Update weights\n\n"
                    "As soon as you return logits or probabilities from `forward(x)`, the predictions will appear below!"
                )
            elif forward_err:
                st.warning(f"⚠️ `forward()` encountered an error during call:\n`{forward_err}`")
                st.info("Check `LinearNeuralNetwork.forward` signature and implementation.")
            elif forward_executed and result is not None:
                output_tensor = result
                if isinstance(result, tuple) and len(result) > 0:
                    output_tensor = result[0]
                elif isinstance(result, dict) and "output" in result:
                    output_tensor = result["output"]
                
                if isinstance(output_tensor, np.ndarray):
                    flat_out = output_tensor.flatten()
                    if flat_out.size == 10:
                        if np.any(flat_out < 0) or not np.isclose(np.sum(flat_out), 1.0, atol=1e-2):
                            exp_vals = np.exp(flat_out - np.max(flat_out))
                            probs = exp_vals / np.sum(exp_vals)
                        else:
                            probs = flat_out
                            
                        predicted_digit = int(np.argmax(probs))
                        confidence = float(probs[predicted_digit]) * 100
                        
                        st.metric(
                            label="Predicted Digit",
                            value=f"{predicted_digit}",
                            delta=f"{confidence:.1f}% confidence",
                        )
                        
                        chart_df = pd.DataFrame({
                            "Digit": [str(i) for i in range(10)],
                            "Probability": probs,
                        })
                        st.bar_chart(chart_df.set_index("Digit"), y="Probability", height=220)
                    else:
                        st.write("Forward output shape:", output_tensor.shape)
                        st.write("Raw output:", output_tensor)
                else:
                    st.write("Forward returned:", output_tensor)

# ---------------------------------------------------------
# Educational Inspection Section: Weights & Activations
# ---------------------------------------------------------
st.divider()
with st.expander("📚 Model Inspection (Weights & Biases in `model`)"):
    st.markdown("Here students can inspect any attributes defined on `self` in `LinearNeuralNetwork`:")
    if model is not None:
        attrs = {
            k: v for k, v in model.__dict__.items()
            if not k.startswith("_")
        }
        if attrs:
            for name, val in attrs.items():
                if isinstance(val, np.ndarray):
                    st.write(f"- **`self.{name}`**: numpy array with shape `{val.shape}`, dtype `{val.dtype}`")
                else:
                    st.write(f"- **`self.{name}`**: `{type(val).__name__}` = `{val}`")
        else:
            st.info("No attributes found on `self` yet. Once you define `self.W` or `self.b` in `__init__`, they will be listed here.")
    else:
        st.write("Model not initialized.")
