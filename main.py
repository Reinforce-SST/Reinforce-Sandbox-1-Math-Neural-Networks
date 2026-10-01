import subprocess
import sys


def main():
    print("🚀 Launching 28x28 Neural Network Streamlit Visualizer...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])


if __name__ == "__main__":
    main()
