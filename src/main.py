import numpy as np

from models import DRA, EqualRA, FreqRA, StakesRA
from simulator import Simulator
from tasks import Memory2AFC


def train_dra() -> None:
    env = Memory2AFC()
    agent = DRA(size=env.n_states).make()
    simulator = Simulator(agent, env)

    print(f"Training {agent.model.value}")

    for _ in range(1770):
        simulator.run_episode()

    print("\nFinished training!\n")


def train_all_models() -> None:
    envs = [Memory2AFC() for _ in range(4)]
    size = envs[0].n_states
    agents = [
        DRA(size).make(),
        FreqRA(size).make(),
        StakesRA(size).make(),
        EqualRA(size).make(),
    ]
    simulators = [Simulator(agent, env) for agent, env in zip(agents, envs)]

    for simulator in simulators:
        print(f"Training {simulator.agent.model.value}")
        for _ in range(1770):
            simulator.run_episode()

    print("\nFinished training!\n")

    # print results
    for agent in agents:
        print(agent.model)
        print(np.around(agent.q_table.values, 0)[:12])
        print(np.around(agent.noise_table.values, 1)[:12])
        print("\n")


if __name__ == "__main__":
    train_all_models()
