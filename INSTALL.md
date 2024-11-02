# Installation Guide for Bumblebee
## Requirements
### Base Requirements

- Python >= 3.8
- PyTorch >= 1.8.0
- xFormers >= 0.0.16
- CUDA-capable GPU (recommended)

### Python Version Support

- Python 3.8
- Python 3.9
- Python 3.10
- Python 3.11

## Installation Methods
1. Using pip from PyPI (Recommended)
```bash
pip install tiled-ring-buffer-attention
```
2. Using pip from GitHub
```bash
pip install git+https://github.com/IAMAl/Bumblebee.git
```
3. Manual Installation
### Clone the Repository
```bash
git clone https://github.com/IAMAl/Bumblebee.git
cd Bumblebee
```
### Basic Installation
```bash
pip install -e .
```
### Development Installation
#### Install with development dependencies:
```bash
pip install -e ".[dev]"
```
### Full Installation
#### Install with all optional dependencies:
```bash
pip install -e ".[dev,docs,benchmark]"
```
## Optional Dependencies
### Development Tools
```bash
pip install -e ".[dev]"
```
Includes:

- pytest: Testing framework
- pytest-cov: Coverage reporting
- black: Code formatting
- isort: Import sorting
- mypy: Static type checking
- flake8: Code linting

## Documentation Tools
```bash
pip install -e ".[docs]"
```
Includes:

- sphinx: Documentation generator
- sphinx-rtd-theme: ReadTheDocs theme
- sphinx-autodoc-typehints: Type hints support

## Benchmarking Tools
```bash
pip install -e ".[benchmark]"
```
Includes:

- pytorch-benchmark-utils
- memory-profiler

## GPU Support
### CUDA Installation

1. Ensure you have a CUDA-capable GPU
2. Install CUDA Toolkit (compatible with your PyTorch version)
3. Install appropriate NVIDIA drivers

### PyTorch with CUDA
Install PyTorch with CUDA support:
```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
### Verification
Verify the installation:
```python
import torch
import tiled_ring_buffer_attention

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

### Common Issues
#### CUDA Issues
If encountering CUDA-related issues:

1. Verify CUDA installation:

```bash
nvidia-smi
```

2. Check PyTorch CUDA compatibility:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```

## xFormers Compatibility
If encountering xFormers issues:

1. Verify xFormers installation:

```python
import xformers
print(xformers.__version__)
````

2. Ensure PyTorch and xFormers versions are compatible

## Development Setup
1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```
2. Install Development Dependencies
```bash
pip install -e ".[dev]"
```
3. Setup Pre-commit Hooks
```bash
pre-commit install
```
4. Running Tests
```bash
pytest tests/
```
5. Code Formatting
```bash
# Format code
black .

# Sort imports
isort .

# Type checking
mypy src/

# Lint code
flake8 src/
```

## Documentation
### Building Documentation
```bash
cd docs
make html
```

Documentation will be available in docs/_build/html/

## Support
For issues and questions:

- GitHub Issues: https://github.com/IAMAl/Bumblebee/issues

## License
BSD 3-Clause License - See LICENSE file for details