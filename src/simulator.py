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

    def run_episode(self) -> Reward:
        # Initializing some variables
        tot_reward = 0
        done = False
        state = self.env.reset()

        while not done:
            experience, done = act_and_step(self.agent, self.env, state)
            self.agent.observe(experience)
            self.agent.update_values(experience)
            self.agent.update_visit_counts(experience)

            # Update state and total reward obtained
            state = experience["next_state"]
            tot_reward += experience["reward"]

        # allocate resources
        self.agent.allocate_memory_resources()

        return tot_reward

    # def update_agent_noise(self):
    #     """Generate trajectories on policy."""

    #     grads = []

    #     for _ in range(self.agent.p.n_trajectories):
    #         # Initializing some variables
    #         grad = np.zeros(self.agent.sigma.shape)
    #         tot_reward, reward = 0, 0
    #         done = False
    #         state = self.env.reset()
    #         r = []

    #         while not done:

    #             exp, done = act_and_step(self.agent, self.env, state)
    #             r.append(exp["reward"])

    #             idx_sa = self.agent._index(state=state, action=exp["action"])
    #             idx_s = self.agent._index(
    #                 state=state, action_space=self.env.action_space
    #             )

    #             # advantage function
    #             psi = self.agent.q[idx_sa] - np.dot(
    #                 self.agent.q[idx_s], exp["prob_actions"]
    #             )

    #             # gradients
    #             grad[idx_s] -= psi * (
    #                 self.agent.p.beta * exp["zeta"] * exp["prob_actions"]
    #             )
    #             grad[idx_sa] += psi * self.agent.p.beta * exp["zeta"][exp["action"]]

    #             # Update state and total reward obtained
    #             state = exp["next_state"]
    #             tot_reward += reward

    #         # collect sampled stoch. gradients for all trajectories
    #         grads += [grad]

    #     # Setting fixed and terminal sigmas to sigma_base to avoid
    #     # divide by zero error; reset to 0 at the end of the loop
    #     self.agent.sigma[12:] = self.agent.p.sigma_base

    #     # Compute average gradient across sampled trajs & cost
    #     grad_cost = (
    #         self.agent.sigma / (self.agent.p.sigma_base**2) - 1 / self.agent.sigma
    #     )
    #     grad_mean = np.mean(grads, axis=0)

    #     # Updating sigmas
    #     self.agent.sigma += self.agent.p.lr * (
    #         grad_mean - self.agent.p.lmda * grad_cost
    #     )

    #     # reset the original state
    #     self.env._episode -= self.agent.p.n_trajectories
    #     self.agent.sigma[12:] = 0

    #     return
