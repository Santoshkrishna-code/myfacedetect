#!/usr/bin/env python3
"""
MyFaceDetect Setup
==================

Setup script for MyFaceDetect face detection and recognition library.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read version
version = "0.3.0"

setup(
    name="myfacedetect",
    version=version,
    author="Santosh Krishna",
    author_email="santoshkrishna.code@gmail.com",
    description="High-performance face detection and recognition library with CPU-only support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Santoshkrishna-code/myfacedetect",
    packages=find_packages(
        exclude=["tests*", "testing*", "scripts*", "examples*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Graphics :: Capture :: Digital Camera",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.5.0",
        "mediapipe>=0.9.0",
        "numpy>=1.21.0",
        "Pillow>=8.0.0",
        "scikit-learn>=1.0.0",
        "ultralytics>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
        ],
        "examples": [
            "matplotlib>=3.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "myfacedetect-demo=myfacedetect.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "myfacedetect": [
            "*.yaml",
            "*.yml",
            "*.json",
            "data/*.xml",
            "models/*.pt",
            "config/*.yaml",
        ],
    },
    zip_safe=False,
    keywords=[
        "face detection",
        "face recognition",
        "computer vision",
        "opencv",
        "mediapipe",
        "cpu-only",
        "real-time",
    ],
    project_urls={
        "Bug Reports": "https://github.com/Santoshkrishna-code/myfacedetect/issues",
        "Source": "https://github.com/Santoshkrishna-code/myfacedetect",
        "Documentation": "https://github.com/Santoshkrishna-code/myfacedetect#readme",
    },
)
