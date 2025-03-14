from setuptools import setup, find_packages

setup(
    name="signal_response_study",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
    ],
    # Optional metadata
    author="Your Name",
    description="An analyzer for end-to-end training and prediction flow.",
    python_requires=">=3.7",  # or whichever Python version you need
)
