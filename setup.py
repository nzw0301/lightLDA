#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import pkg_resources
from setuptools import setup, find_packages
import os
import codecs
import re
import sys

def read(*parts):
    path = os.path.join(os.path.dirname(__file__), *parts)
    with codecs.open(path, encoding='utf-8') as fobj:
        return fobj.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

install_requires = [
    'numpy'
]

setup(
    name='lightlda',
    version=find_version("lightlda", "__init__.py"),
    description='fast sampling algorithm based on CGS',
    author='nzw0301 ',
    author_email='',
    license='MIT',
    keywords='NLP,sampling,algorithm,lightLDA,topic-modeling',
    url='https://github.com/nzw0301/lightLDA',
    install_requires=install_requires,
    packages = find_packages(),
    package_dir={'lightlda': 'lightlda'},
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3'
      ]
)