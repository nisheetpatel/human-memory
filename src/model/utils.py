from dataclasses import dataclass

from definitions import Action, State


@dataclass
class IndexingError(Exception):
    error: str


def option_choice_set_2afc(state: State) -> list:
    """Returns choice set for Memory_2AFC task."""
    if state < 12:
        if state % 3 == 0:
            choice_set = [state + 1, state + 2]  # 1 v 2; PMT 0
        elif state % 3 == 1:
            choice_set = [state - 1, state + 1]  # 0 v 2; PMT 1
        else:
            choice_set = [state - 2, state - 1]  # 0 v 1; PMT 2
    elif state < 24:
        choice_set = [state - 12, state]
    elif state < 36:
        choice_set = [state - 24, state]
    return choice_set


def indexer_2afc(state: State = None, action: Action = None):
    """Returns q-table index entry for the Memory 2AFC task."""
    if state == -1:
        return state
    if (state is not None) & (action is not None):
        return option_choice_set_2afc(state)[action]
    if state is not None:
        return option_choice_set_2afc(state)
    raise IndexingError("Fatal: no cases match for indexer.")


def option_choice_set_slots(state: State) -> list:
    """Returns the choice set for Slot Machines task."""
    return [state // 4, state % 4 + 4]


def indexer_slots(state: State = None, action: Action = None):
    """Returns q-table index entry for the Memory 2AFC task."""
    if state == -1:
        return state
    if (state is not None) & (action is not None):
        return option_choice_set_slots(state)[action]
    if state is not None:
        return option_choice_set_slots(state)
    raise IndexingError("Fatal: no cases match for indexer.")


@dataclass
class ModelParams:
    sigma_base: float = 5
    gamma: float = 1
    beta: float = 10
    lmda: float = 0.075
    lr: float = 0.01
    n_trajectories: int = 10


@dataclass
class SlotTaskParams:
    delta: float = 0.75
    sigma: float = 5
    rel_freq: float = 3
    rel_diff: float = 3
