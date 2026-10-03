import struct
import numpy as np

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
