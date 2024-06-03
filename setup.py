#!/usr/bin/env python

from setuptools import setup
import sys
try:
    import tomllib as toml
except ModuleNotFoundError:
    try:
        import toml
    except ModuleNotFoundError:
        print("Outdated Python version detected.\n"
              "Please install 'toml' to continue: "
              " pip3 install toml")
        sys.exit(1)


METADATA_FILE = "pyproject.toml"


def readme():
    with open('README.md') as f:
        return f.read()


mode = 'rb' if sys.version_info >= (3, 11) else 'r'
with open(METADATA_FILE, mode) as fb:
    metadata = toml.load(fb)


setup(
    name=metadata['project']['name'],
    version=metadata['project']['version'],
    author=metadata['project']['authors'][-1]['name'],
    author_email=metadata['project']['authors'][-1]['email'],
    url=metadata['project']['urls']['Homepage'],
    install_requires=metadata['project']['dependencies'],
    description=metadata['project']['description'],
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    license='GLP3',
    keywords=metadata['project']['keywords'],
    python_requires=metadata['project']['requires-python'],
)
