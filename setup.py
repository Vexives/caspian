from setuptools import setup, find_packages
from caspian import __version__


setup(
    name = "caspian-ml",
    version = __version__,
    license = "Apache 2.0",
    description = "A deep learning library focused entirely around NumPy.",
    long_description = open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    author = "Vexives",
    url="https://github.com/Vexives/caspian",
    packages = find_packages(),
    install_requires = ["numpy"],
    test_suite = "tests",
    tests_require = ["pytest"],
    python_requires = ">=3.10",
    keywords = [
        "machine learning", "data science", "deep learning", "numpy"
    ],
    classifiers = [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries",
        "Operating System :: OS Independent"
    ]
)