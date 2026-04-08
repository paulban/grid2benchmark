"""Algorithm template for grid2benchmark.

Implement build_agent(env, context) returning an agent with an act(observation) method.
"""

from __future__ import annotations


class TemplateAgent:
    """Minimal no-op agent for validation runs."""

    def __init__(self, action_space):
        """Store Grid2Op action space for no-op action creation."""
        self._action_space = action_space

    def act(self, observation):
        """Return a no-op action for the current observation."""
        # Return a baseline no-op action.
        return self._action_space()


def build_agent(env, context):
    """Return an agent instance.

    Args:
        env: Grid2Op environment (or EnvRecorder wrapping one).
        context: Dict with benchmark and scenario metadata.
            Example keys:
                context["benchmark"]["max_steps"]
                context["benchmark"]["kpis"]
                context["scenario"]["env_name"]
                context["scenario"]["topology"]
                context["scenario"]["time_series"]
                context["scenario"]["time_series_ids"]
    """
    _ = context
    return TemplateAgent(env.action_space)
