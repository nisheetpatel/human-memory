from typing import TypedDict

# custom type aliases
Action = int
ActionSpace = list[int, int]
State = int
Reward = float
Done = bool
Info = dict


class Experience(TypedDict):
    state: State
    action: Action
    reward: Reward
    next_state: State
    prob_actions: list[float]
