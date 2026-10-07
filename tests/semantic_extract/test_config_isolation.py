"""The process-wide semantic_extract config does not carry state between tests.

The tests in this file run in order: each odd one leaves something behind the
way a careless test would, and the next one checks it is gone (see #1913).
"""

import sys
from unittest.mock import MagicMock

from semantica.semantic_extract.config import config


def test_leaves_a_double_in_the_config():
    config.set_provider("openai", api_key=MagicMock())
    assert isinstance(config.get_api_key("openai"), MagicMock)


def test_does_not_see_the_double_left_by_the_previous_test():
    assert not isinstance(config.get_api_key("openai"), MagicMock)


def test_leaves_a_stand_in_config_module_in_sys_modules():
    sys.modules["semantica.semantic_extract.config"] = MagicMock()


def test_does_not_see_the_stand_in_config_module():
    assert not isinstance(sys.modules["semantica.semantic_extract.config"], MagicMock)
