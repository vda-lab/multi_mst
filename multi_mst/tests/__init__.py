import os
import pytest

def run_tests():
    pytest.main([os.sep.join(__file__.split(os.sep)[:-1])])