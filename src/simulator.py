from dataclasses import dataclass

from customtypes import Done, Experience, Reward, State
from models import Agent, NoisyQAgent
from tasks import Environment


def act_and_step(
    agent: Agent, env: Environment, state: State
) -> tuple[Experience, Done]:
    """Agent takes an action, environment steps."""
    # Determine next action
    action, prob_actions, zeta = agent.act(state)

    # Get next state and reward
    next_state, reward, done, _ = env.step(action)

    # setup experience dict
    experience = {
        "state": state,
        "action": action,
        "reward": reward,
        "next_state": next_state,
        "zeta": zeta,
        "choice_set": env.choice_set,
        "prob_actions": prob_actions,
    }

    return (experience, done)


@dataclass
class Simulator:
    agent: NoisyQAgent
    env: Environment

    def run_episode(self, learning=True) -> Reward:
        # Initializing some variables
        tot_reward = 0
        done = False
        state = self.env.reset()

        while not done:
            experience, done = act_and_step(self.agent, self.env, state)
            self.agent.observe(experience)
            if learning:
                self.agent.update_values(experience)

            # Update state and total reward obtained
            state = experience["next_state"]
            tot_reward += experience["reward"]

        # allocate resources
        if learning:
            self.agent.allocate_memory_resources()

        return tot_reward

    def train_agent(self) -> None:
        for i in range(self.env.n_episodes):  # pylint: disable=no-member
            learning = True
            if i > self.env.n_episodes - 2 * 192:  # pylint: disable=no-member
                learning = False
            self.run_episode(learning)
