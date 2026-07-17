#!/usr/bin/env python
"""Setup script for DeCIR package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().strip().split("\n")
requirements = [r.strip() for r in requirements if r.strip() and not r.startswith("#")]

setup(
    name="decir",
    version="0.1.0",
    author="DeCIR Authors",
    author_email="your.email@example.com",
    description="Dual-modal Semantic Decoupling for Composed Image Retrieval",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/DeCIR",
    packages=find_packages(exclude=["tests", "scripts", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=4.0",
            "isort>=5.10",
        ],
    },
    entry_points={
        "console_scripts": [
            "decir-infer=decir.cli:main",
        ],
    },
)
