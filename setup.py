#!/usr/bin/env python

from setuptools import setup


version = '0.1.0'

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='hypodisc',
    version=version,
    author='Xander Wilcke',
    author_email='w.x.wilcke@vu.nl',
    url='https://wxwilcke.gitlab.io/hypodisc',
    install_requires=['pyRDF >= 2.2.1', 'numpy', 'scikit-learn'],
    download_url = 'https://gitlab.com/wxwilcke/hypodisc/-/archive/' + version + '/hypodisc-' + version + '.tar.gz',
    description='Hypothesis Discovery on RDF Knowledge Graphs',
    long_description = open('README.md').read(),
    long_description_content_type="text/markdown",
    license='GLP3',
    include_package_data=True,
    zip_safe=True,
    keywords = ["rdf", "knowledge graphs", "pattern discovery", "hypothesis generation"],
    python_requires='>=3.9',
)
