"""Project-specific constants for the multimodal Ant server."""

from enum import Enum

DEFAULT_CLIENT_NAME = "unknown"
MAX_STEPS = 1200
ACTION_DIMENSION = 8
NO_OP_ACTION = [0.0] * ACTION_DIMENSION


class AntTrajectoryColumn(str, Enum):
    """Column names for trajectory CSV recording."""

    TORSO_X = "torso_x"
    TORSO_Y = "torso_y"
    GOALS_ACHIEVED = "goals_achieved"
