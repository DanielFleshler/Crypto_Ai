#!/usr/bin/env python3
"""
Setup script for Crypto Bot Trader project.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Get version
def get_version():
    version_file = os.path.join("trading_strategy", "__init__.py")
    if os.path.exists(version_file):
        with open(version_file, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "1.0.0"

setup(
    name="crypto-bot-trader",
    version=get_version(),
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced cryptocurrency trading bot with Elliott Wave and ICT analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/crypto-bot-trader",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/crypto-bot-trader/issues",
        "Source": "https://github.com/yourusername/crypto-bot-trader",
        "Documentation": "https://github.com/yourusername/crypto-bot-trader/docs",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "pytest-xdist>=3.3.0",
            "pytest-html>=3.2.0",
            "pytest-benchmark>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "isort>=5.12.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "performance": [
            "memory-profiler>=0.61.0",
            "psutil>=5.9.0",
            "tqdm>=4.65.0",
            "line-profiler>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "crypto-bot-trader=trading_strategy.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "trading_strategy": [
            "config/*.yaml",
            "data/*.csv",
            "data/*.parquet",
        ],
    },
    zip_safe=False,
    keywords=[
        "cryptocurrency",
        "trading",
        "bot",
        "elliott-wave",
        "ict",
        "market-structure",
        "technical-analysis",
        "backtesting",
        "risk-management",
    ],
    license="MIT",
    platforms=["any"],
    test_suite="tests",
    tests_require=[
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-mock>=3.11.0",
    ],
)
