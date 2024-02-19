from .models import DDRA, Agent
from .task import Env, SlotMachinesTask


class Simulator:

    def __init__(self, env: Env = SlotMachinesTask, agent: Agent = DDRA(), n_episodes: int = 1_000):
        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes

    def run_episode(self) -> tuple[int, float, float, bool]:
        sm_id, price = self.env.reset()
        action = self.agent.act(sm_id, price)
        _, reward, _, _ = self.env.step(action)
        self.agent.update(sm_id, price, reward, action)
        return (sm_id, price, reward, action)

    def train_agent(self, record_data: bool = False) -> list[tuple]:
        data = []
        for _ in range(self.n_episodes):
            data_tuple = self.run_episode()
            if record_data:
                data.append(data_tuple)
        return data

