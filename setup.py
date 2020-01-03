#!/usr/bin/env python
# -*- coding: utf-8 -*-
import setuptools
import re


with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()


with open('deepdanbooru/__main__.py', encoding='utf-8') as f:
    version = re.search('__version__ = \'([^\']+)\'', f.read()).group(1)  # type: ignore


setuptools.setup(
    name="DeepDanbooru",
    version=version,
    author="Kichang Kim",
    author_email="admin@kanotype.net",
    description="AI based multi-label girl image classification system, "
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
    install_requires=[
        'Click==7.0',
        'numpy>=1.16.2+mkl',
        'scikit-image==0.15.0',
        'requests>=2.22.0',
    ],
    extras_require={
        'tensorflow': ['tensorflow==2.1.0rc1'],
        'test': ['pytest', 'flake8', 'mypy']
    },
    entry_points={
        "console_scripts": [
            "deepdanbooru=deepdanbooru.__main__:main",
        ]
    },
)
