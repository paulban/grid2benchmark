"""TorchRandomAgent algorithm example.

Simulates a neural network policy using a torch model. The agent builds a random model and looks for a model file in
the "model" directory next to this script to load pre-defined weights. If no model file is found, the random model
itself is used to simulate inference. This agent is intended only to simulate action building from a neural network.

Note: To ensure valid actions are returned, the agent samples from the action space rather than using model outputs.
"""

from __future__ import annotations

from pathlib import Path

import os

import numpy as np
import torch
import torch.nn as nn

# Use None to omit loading the weights from a model file
MODEL_FILE: str | None = os.path.join("model", "l2rpn_case14_sandbox_torch_random_agent_model.pt")


def _build_model(obs_size: int) -> nn.Sequential:
    """Return a simple neural network model for the given observation size.

    :param obs_size: Observation size.
    :return: Neural network model.
    """
    return nn.Sequential(
        nn.Linear(obs_size, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )


class TorchRandomAgent:
    """Agent that uses a torch model for inference.

    This agent is intended only to simulate action building from a neural network. To ensure valid actions are returned,
    the agent samples from the action space rather than using model outputs.

    :param obs_size: Observation size.
    :param action_space: Action space.
    :param model_file: Model file.
    """

    def __init__(self, obs_size: int, action_space, model_file: str  | Path | None = None):
        self._action_space = action_space

        model = _build_model(obs_size)

        # If a model file is provided, simulate loading weights
        if model_file is not None and Path(model_file).exists():
            model.load_state_dict(torch.load(model_file, weights_only=True))

        model.eval()
        self._model = model

    def act(self, observation, reward: float = 0.0, done: bool = False):
        """Act method for the agent.

        :param observation: Observation in numpy format.
        :param reward: Reward. Unused.
        :param done: Flag indicating whether the environment is done or not. Unused.
        :return: Action.
        """
        obs_vec = observation.to_vect().astype(np.float32)
        obs_tensor = torch.from_numpy(obs_vec).unsqueeze(0)

        with torch.no_grad():
            # Simulate inference run using the model
            _ = self._model(obs_tensor)

        # To ensure we have valid actions, return a random action from the action space
        return self._action_space.sample()


def build_agent(env, context):
    """Build the agent.

    :param env: Environment.
    :param context: Context.
    :return: Agent instance.
    """
    _ = context
    obs_size = env.observation_space.size()
    return TorchRandomAgent(obs_size, env.action_space, model_file=MODEL_FILE)
