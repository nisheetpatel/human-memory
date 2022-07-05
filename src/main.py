import numpy as np

from models import DRA, EqualRA, FreqRA, StakesRA
from simulator import Simulator
from tasks import Memory2AFC


def train_dra() -> float:
    env = Memory2AFC()
    agent = DRA(q_size = env.n_states)
    simulator = Simulator(agent, env)
    reward = simulator.run_episode()
    print(f"Reward obtained = {round(reward,2)}")
    return reward

def train_all_models() -> None:
    envs = [Memory2AFC() for _ in range(4)]
    q_size = envs[0].n_states
    agents = [DRA(q_size), FreqRA(q_size), StakesRA(q_size), EqualRA(q_size)]
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
