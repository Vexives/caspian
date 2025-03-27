from setuptools import setup, find_packages
from . import __version__

setup(
    name = "Caspian",
    version = __version__,
    description = "A deep learning library focused entirely around NumPy.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author = "Vexives",
    packages = find_packages(),
    install_requires = ["numpy"],
    python_requires = ">=3.10"
)