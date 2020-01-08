#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re

import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()


with open('deepdanbooru/__main__.py', encoding='utf-8') as f:
    version = re.search('__version__ = \'([^\']+)\'', f.read()).group(1)  # type: ignore


install_requires = [
    'Click>=7.0',
    'numpy>=1.16.2',
    'scikit-image>=0.15.0',
    'requests>=2.22.0',
    'six>=1.13.0',
]
tensorflow_pkg = 'tensorflow>=2.1.0'

setuptools.setup(
    name="deepdanbooru",
    version=version,
    author="Kichang Kim",
    author_email="admin@kanotype.net",
    description="DeepDanbooru is AI based multi-label girl image classification system, "
    "implemented by using TensorFlow.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KichangKim/DeepDanbooru",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=install_requires,
    extras_require={
        'tensorflow': [tensorflow_pkg],
        'test': ['pytest', 'flake8', 'mypy']
    },
    entry_points={
        "console_scripts": [
            "deepdanbooru=deepdanbooru.__main__:main",
        ]
    },
)
