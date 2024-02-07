#!/usr/bin/env python

from setuptools import setup
import toml


METADATA_FILE = "pyproject.py"


def readme():
    with open('README.md') as f:
        return f.read()


metadata = toml.load(METADATA_FILE)


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
