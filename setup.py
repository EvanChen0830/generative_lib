from setuptools import setup, find_packages

setup(
    name="generative_lib",
    version="0.1.0",
    description="A modular library for Generative Models (Diffusion, Flow Matching, Consistency Models)",
    author="Zong Yu Chen",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "mlflow",
        "tqdm",  # For progress bars in trainers
    ],
    extras_require={
        "dev": [
            "matplotlib",
            "scikit-learn",
            "pytest"
        ]
    }
)
