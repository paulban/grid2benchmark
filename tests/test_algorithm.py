"""Unit tests for algorithm loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from grid2benchmark._algorithm import (
    load_algorithm,
    load_algorithm_from_file,
    validate_algorithm,
)


VALID_SRC = """
class _Agent:
    def __init__(self, action_space):
        self._action_space = action_space
    def act(self, observation):
        return self._action_space()

def build_agent(env, context):
    return _Agent(env.action_space)
"""

MISSING_BUILD_AGENT_SRC = """
class Foo:
    pass
"""

NOT_CALLABLE_BUILD_AGENT_SRC = """
build_agent = "not_a_function"
"""


class TestLoadAlgorithm:
    def test_valid_source_returns_module(self):
        module = load_algorithm(VALID_SRC)
        assert hasattr(module, "build_agent")

    def test_build_agent_is_callable(self):
        module = load_algorithm(VALID_SRC)
        assert callable(module.build_agent)

    def test_syntax_error_raises(self):
        with pytest.raises(SyntaxError):
            load_algorithm("def oops(:\n    pass")

    def test_empty_source_loads_without_build_agent(self):
        module = load_algorithm("")
        assert not hasattr(module, "build_agent")


class TestLoadAlgorithmFromFile:
    def test_loads_template_file(self, algorithm_template_path: Path):
        module = load_algorithm_from_file(algorithm_template_path)
        assert hasattr(module, "build_agent")

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_algorithm_from_file(tmp_path / "nonexistent.py")

    def test_loads_greedy_baseline_file(self):
        path = Path(__file__).parent.parent / "examples" / "greedy_baseline.py"
        module = load_algorithm_from_file(path)
        assert callable(module.build_agent)


class TestValidateAlgorithm:
    def test_valid_module_passes(self):
        module = load_algorithm(VALID_SRC)
        validate_algorithm(module)  # must not raise

    def test_missing_build_agent_raises(self):
        module = load_algorithm(MISSING_BUILD_AGENT_SRC)
        with pytest.raises(ValueError, match="build_agent"):
            validate_algorithm(module)

    def test_non_callable_build_agent_raises(self):
        module = load_algorithm(NOT_CALLABLE_BUILD_AGENT_SRC)
        with pytest.raises(ValueError, match="build_agent"):
            validate_algorithm(module)
