# -*- coding: utf-8 -*-

import pytest
from adl_vod_encoder.skeleton import fib

__author__ = "lmoesing"
__copyright__ = "lmoesing"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
