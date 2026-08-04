# Environment

- Python: 3.11.7
- PyTorch: 2.13.0
- NumPy: 2.4.6
- pandas: 3.0.5
- Matplotlib: 3.11.1
- pytest: 9.1.1
- PyArrow: 25.0.0
- OS: macOS 26.3 (build 25D125), Darwin 25.3.0
- Machine: MacBook Pro (Mac17,2), Apple M5, arm64
- Device: CPU only

The environment was created with `python3 -m venv census/.venv`. PyArrow is the
only dependency beyond the five requested packages; it is needed as pandas' engine
for writing the required Parquet artifact. No TensorFlow, Julia, MPS, GPU, Docker,
or Rosetta environment was used. All model tensors and computations are explicitly
kept on `torch.device("cpu")`; float64 is used for threshold computation.
