import os
from enum import Enum
from typing import TypedDict

ROOT_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
DATA_PATH = os.path.join(ROOT_DIR, "data/")
MODEL_PATH = os.path.join(ROOT_DIR, "models/")
FIGURE_PATH = os.path.join(ROOT_DIR, "figures/")


# custom type aliases
Action = int
State = int
Reward = float
Done = bool
Info = dict


class ModelName(Enum):
    DRA = "DRA"
    FREQ = "Frequency"
    STAKES = "Stakes"
    EQUALPRECISION = "Equal-Precision"


class Experience(TypedDict):
    state: State
    action: Action
    reward: Reward
    next_state: State
    zeta: list[float]
    action_idx: int
    prob_actions: list[float]


ExperienceBuffer = list[Experience]
