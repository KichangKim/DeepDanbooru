#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib
from unittest import mock

import pytest
from click.testing import CliRunner


def test_import():
    import deepdanbooru


@pytest.mark.parametrize('func_name', ['main', 'evaluate'])
def test_help(func_name):
    mod = importlib.import_module('deepdanbooru.__main__')
    runner = CliRunner()
    result = runner.invoke(getattr(mod, func_name), ['--help'])
    assert result.exit_code == 0
    assert result.output


@pytest.fixture
def packages():
    with open('requirements.txt') as f:
        return f.read().splitlines()


def test_package_setup(packages):
    import setuptools
    with mock.patch('setuptools.setup'):
        import setup
        setup_pkgs = setup.install_requires
        tensorflow_pkg = setup.tensorflow_pkg
    assert setup_pkgs == list(
        filter(lambda x: not x.startswith('tensorflow'), packages))
    assert list(
        filter(lambda x: x.startswith('tensorflow'), packages)
    ) == [tensorflow_pkg]


def test_readme_pkg(packages):
    with open('README.md') as f:
        text = f.read()
    line_start = 'Following packages are need to be installed.'
    line_end = '\n\n'
    pkg_text = text.split(line_start)[1].rsplit(line_end)[0].strip()
    readme_pkgs = list(map(lambda x: x.split(' ', 1)[1], pkg_text.splitlines()))

    assert list(
        sorted(filter(lambda x: not x.startswith('tensorflow'), readme_pkgs))
    ) == list(
        sorted(filter(lambda x: not x.startswith('tensorflow'), packages))
    )

    assert list(
        filter(lambda x: x.startswith('tensorflow'), packages)
    ) == list(
        filter(lambda x: x.startswith('tensorflow'), readme_pkgs)
    )
