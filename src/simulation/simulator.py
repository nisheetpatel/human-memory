import pandas as pd

from src.simulation.models import DRA, Agent
from src.simulation.task import (
    Env,
    SlotMachinesTask,
    SlotMachinesTaskWithPredefinedData,
)


class Simulator:
    def __init__(
            self,
            env: Env = SlotMachinesTask(),
            agent: Agent = DRA(),
            n_episodes: int = 10_000,
            predefined_data: list[tuple[int, float, float]] = None
        ) -> None:
        
        self.agent = agent

        if predefined_data is None:
            self.env = env
            self.n_episodes = n_episodes
        else:
            self.env = SlotMachinesTaskWithPredefinedData(predefined_data)
            self.n_episodes = len(predefined_data) - 1

    def run_episode(self) -> tuple[int, float, float, float, bool]:
        sm_id, price = self.env.reset()
        action = self.agent.act(sm_id, price)
        _, reward, _, info = self.env.step(action)
        self.agent.update(sm_id, price, reward, info["return"], action)
        return sm_id, price, reward, action

    def train_agent(self, record_data: bool = False) -> pd.DataFrame:
        data = []
        for _ in range(self.n_episodes):
            data_tuple = self.run_episode()
            if record_data:
                data.append(data_tuple)
        columns = ["sm_id", "price", "reward", "action"]
        self.data = pd.DataFrame.from_records(data, columns=columns)
        return self.data

