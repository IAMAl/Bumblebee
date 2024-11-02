from setuptools import setup, find_packages
import os

# Read requirements.txt
def read_requirements(filename="requirements.txt"):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tiled_ring_buffer_attention",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Core dependencies
    install_requires=[
        "torch>=1.8.0",
        "xformers>=0.0.16",
        "numpy>=1.20.0",
        "einops>=0.6.0",       # For tensor operations
        "typing_extensions>=4.0.0",  # For Python 3.8 compatibility
    ],
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",  # For coverage reports
            "black>=22.0.0",      # Code formatting
            "isort>=5.10.0",      # Import sorting
            "mypy>=0.990",        # Type checking
            "flake8>=4.0.0",      # Code linting
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.12.0",
        ],
        "benchmark": [
            "pytorch-benchmark-utils>=0.1.0",
            "memory-profiler>=0.60.0",
        ],
    },
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Metadata
    author="IAMAl",
    author_email="refactoring.day@gmail.com",
    description="A memory-efficient attention implementation with ring buffers and hierarchical tiled computation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IAMAl/Bumblebee",
    project_urls={
        "Bug Tracker": "https://github.com/IAMAl/Bumblebee/issues",
        "Documentation": "https://bumblebee.readthedocs.io/",
        "Source Code": "https://github.com/IAMAl/Bumblebee",
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Framework :: Pytorch",
        "Operating System :: OS Independent",
    ],
    
    # Package data
    include_package_data=True,
    package_data={
        "tiled_ring_buffer_attention": [
            "py.typed",             # For PEP 561 compliance
            "*.pyi",                # Type stubs
        ],
    },
    
    # Entry points
    entry_points={
        "console_scripts": [
            "tiled-ring-buffer=tiled_ring_buffer_attention.cli:main",
        ],
    },
    
    # Keywords for PyPI
    keywords=[
        "attention",
        "transformer",
        "deep-learning",
        "machine-learning",
        "pytorch",
        "xformers",
        "memory-efficient",
        "ring-buffer",
        "tiled-computation",
    ],
)