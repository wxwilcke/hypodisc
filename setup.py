#!/usr/bin/env python

from setuptools import find_namespace_packages, setup
from setuptools.command.install import install
import os


def readme():
    with open('README.md') as f:
        return f.read()

class PostInstallCommand(install):
    def run(self):
        install.run(self)
        os.system("cat install.txt")

setup(
    name='hypodisc',
    version='0.1',
    author='Xander Wilcke',
    author_email='w.x.wilcke@vu.nl',
    url='https://gitlab.com/wxwilcke/hypodisc',
    description='Deep Hypothesis Discovery on RDF Knowledge Graphs',
    license='GLP3',
    include_package_data=True,
    zip_safe=True,
    install_requires=[
        'pybind11',
        'torch',
        'numpy',
        'pillow',
        'pandas',
        'rdflib',
        'rdflib_hdt',
        'matplotlib',
        'deep_geometry',
        'h5py',
        'huggingface_hub',
        'tokenizers',
        'sentencepiece',
        'sacremoses',
        'importlib_metadata'
    ],
    cmdclass={
        'install': PostInstallCommand,
    },
    packages=find_namespace_packages(include=['hypodisc.*']),
)
