from setuptools import setup, find_packages

setup(
    name="tiled_ring_buffer",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.8.0",
        "xformers",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
            "mypy",
        ]
    },
    python_requires=">=3.8",
    author="IAMAl",
    author_email="refactoring.day@gmail.com",
    description="Memory-efficient attention with ring buffers and tiled computation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/IAMAl/Bumblebee",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD 3-Clause License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)