#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib

import pytest
from click.testing import CliRunner


def test_import():
    import deepdanbooru


@pytest.mark.parametrize('func_name', ['main', 'evaluate_images'])
def test_help(func_name):
    mod = importlib.import_module('deepdanbooru.__main__')
    runner = CliRunner()
    result = runner.invoke(getattr(mod, func_name), ['--help'])
    assert result.exit_code == 0
    assert result.output
