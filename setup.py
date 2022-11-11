#!/usr/bin/env python

from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


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
        'torch',
        'numpy',
        'pillow',
        'pandas',
        'rdflib',
        'rdflib_hdt',
        'deep_geometry',
        'h5py',
        'huggingface_hub',
        'tokenizers',
        'sentencepiece',
        'sacremoses',
        'importlib_metadata'
    ],
    packages=['hypodisc'],
)
