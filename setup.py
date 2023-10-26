import codecs
from setuptools import setup, find_packages

setup(
    name="prcskr",
    version="0.1",
    description="An implementation of an algorithms for filtration of data labeling and classification of clients",
    packages=find_packages("srcprcskr"),
    package_dir={"": "srcprcskr"},
    url="https://gitlab.appliedai.tech/priceseekers/core/priceseekers",
    author="Skoltech",
    install_requires=["gpytorch>=1.2.1", "torch", "scikit-learn"],
    python_requires=">=3.6",
)