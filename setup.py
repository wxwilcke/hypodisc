#!/usr/bin/env python

from setuptools import setup
import tomllib


METADATA_FILE = "pyproject.toml"


def readme():
    with open('README.md') as f:
        return f.read()


with open(METADATA_FILE, 'rb') as fb:
    metadata = tomllib.load(fb)


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
