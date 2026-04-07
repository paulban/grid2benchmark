"""Shared pytest fixtures for grid2benchmark tests."""

from __future__ import annotations

from pathlib import Path

import pytest


NOOP_AGENT_SRC = """
class _NoopAgent:
    def __init__(self, action_space):
        self._action_space = action_space

    def act(self, observation):
        return self._action_space()


def build_agent(env, context):
    return _NoopAgent(env.action_space)
"""

MISSING_BUILD_AGENT_SRC = """
class Foo:
    pass
"""

OBSERVABLE_AGENT_SRC = """
class _ObservableAgent:
    def __init__(self, action_space, context):
        self._action_space = action_space
        self.received_context = context

    def act(self, observation):
        return self._action_space()


_last_context = None

def build_agent(env, context):
    global _last_context
    _last_context = context
    return _ObservableAgent(env.action_space, context)
"""


@pytest.fixture()
def noop_agent_source() -> str:
    return NOOP_AGENT_SRC


@pytest.fixture()
def missing_build_agent_source() -> str:
    return MISSING_BUILD_AGENT_SRC


@pytest.fixture()
def algorithm_template_path() -> Path:
    return Path(__file__).parent.parent / "examples" / "algorithm_template.py"
