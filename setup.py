from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="loc2vec",
    version="0.0.1",
    description="Pytorch implementation of the loc2vec method for learning semantic classifications of locations",
    long_description=long_description,
    long_description_content_type="text/markdown",    
    url="https://github.com/angus-spence/loc2vec",
    author="Angus Spence",
    author_email="gusmalcolmspence@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3"
        "Programming Language :: Python :: 3.7"
        "Programming Language :: Python :: 3.8"
        "Programming Language :: Python :: 3.9"
        "Programming Language :: Python :: 3.10"
        "Programming Language :: Python :: 3 :: Only"
    ],
    packages=find_packages(where="src"),
    python_requires=">=3.7, <4"
)