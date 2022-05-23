from enum import Enum
from typing import TypedDict

# custom type aliases
Action = int
ActionSpace = list[int, int]
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
    action_space: ActionSpace
    prob_actions: list[float]


ExperienceBuffer = list[Experience]
