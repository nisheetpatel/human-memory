import numpy as np
import pandas as pd

"""
Defines the class environment as per Huys-Dayan-Rosier's planning task
& the agent's long-term tabular memories: (s,a), r(s,a), Q(s,a), pi(a|s).
"""
class HuysTask:
    # @transition_matrix:   State transition matrix
    # @reward_matrix:       Rewards corresponding to state transitions
    # @n_states:            Number of states
    # @states:              State indices
    def __init__(self, depth=3, n_states=6):
        self.depth = depth
        transitions = np.mat('0 1 0 1 0 0;\
                            0 0 1 0 1 0;\
                            0 0 0 1 0 1;\
                            0 1 0 0 1 0;\
                            1 0 0 0 0 1;\
                            1 0 1 0 0 0')
        rewards = np.mat('0   140  0   20  0   0; \
                          0   0   -20  0  -70  0; \
                          0   0    0  -20  0  -70;\
                          0   20   0   0  -20  0; \
                         -70  0    0   0   0  -20;\
                         -20  0    20  0   0   0')

        # Setting up the transitions and rewards matrices for the
        # extended state space: 6 -> 6 x T_left
        self.transition_matrix = np.zeros(((depth+1)*n_states,(depth+1)*n_states),dtype=int)
        self.reward_matrix = np.zeros(((depth+1)*n_states,(depth+1)*n_states),dtype=int)

        nrows = transitions.shape[0]
        Nrows = self.transition_matrix.shape[0]

        for i in range(nrows,Nrows,nrows):
            self.transition_matrix[i-nrows:i,i:i+nrows] = transitions

        for i in range(nrows,Nrows,nrows):
            self.reward_matrix[i-nrows:i,i:i+nrows] = rewards

        # Transitions, rewards, states
        self.n_states = len(self.transition_matrix)
        self.states = np.arange(self.n_states)
        self.transitions = np.transpose(self.transition_matrix.nonzero())
        self.rewards = np.array(self.reward_matrix[self.reward_matrix != 0])


"""
Custom-made two-step T-maze with 14 states.
"""
class Tmaze:
    # @transition_matrix:   State transition matrix
    # @reward_matrix:       Rewards corresponding to state transitions
    # @n_states:            Number of states
    # @states:              State indices
    def __init__(self, depth=3, n_states=6, gridworld=False):
        self.depth = depth
        self.gridworld = gridworld
        self.transition_matrix = \
            np.mat('0 1 0 0 0 0 0 0 0 0 0 0 0 0;\
                    1 0 1 1 0 0 0 0 0 0 0 0 0 0;\
                    0 1 0 0 1 0 0 0 0 0 0 0 0 0;\
                    0 1 0 0 0 1 0 0 0 0 0 0 0 0;\
                    0 0 1 0 0 0 1 0 0 0 0 0 0 0;\
                    0 0 0 1 0 0 0 1 0 0 0 0 0 0;\
                    0 0 0 0 1 0 0 0 1 0 0 0 0 0;\
                    0 0 0 0 0 1 0 0 0 1 0 0 0 0;\
                    0 0 0 0 0 0 1 0 0 0 1 0 1 0;\
                    0 0 0 0 0 0 0 1 0 0 0 1 0 1;\
                    0 0 0 0 0 0 0 0 1 0 0 0 0 0;\
                    0 0 0 0 0 0 0 0 0 1 0 0 0 0;\
                    0 0 0 0 0 0 0 0 1 0 0 0 0 0;\
                    0 0 0 0 0 0 0 0 0 1 0 0 0 0')
        self.reward_matrix       = -self.transition_matrix
        self.reward_matrix[8,10] =  10
        self.reward_matrix[8,12] =  -5
        self.reward_matrix[9,11] = -10
        self.reward_matrix[9,13] =  5

        # Transitions, rewards, states
        self.n_states = len(self.transition_matrix)
        self.states = np.arange(self.n_states)
        self.transitions = np.transpose(self.transition_matrix.nonzero())
        self.rewards = np.array(self.reward_matrix[self.reward_matrix != 0])


""" 
A wrapper class for a maze, containing all the information about the maze.
Initialized to the 2D maze used by Mattar & Daw 2019 by default, however, 
it can be easily adapted to any other maze by redefining obstacles and size.
"""
class Maze:
    def __init__(self):
        # maze width
        self.WORLD_WIDTH = 9

        # maze height
        self.WORLD_HEIGHT = 6

        # all possible actions
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_LEFT = 2
        self.ACTION_RIGHT = 3
        self.actions = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT]

        # start state
        self.START_STATE = [2, 0]

        # goal state
        self.GOAL_STATES = [[0, 8]]
        self.new_goal = [[0, 6]]

        # all obstacles
        self.obstacles = [[1, 2], [2, 2], [3, 2], [0, 7], [1, 7], [2, 7], [4, 5]]
        self.old_obstacles = [[0,7]]
        self.new_obstacles = None

        # time to change environment
        self.switch_time = 3000

        # initial state action pair values
        # self.stateActionValues = np.zeros((self.WORLD_HEIGHT, self.WORLD_WIDTH, len(self.actions)))

        # the size of q value
        self.q_size = (self.WORLD_HEIGHT, self.WORLD_WIDTH, len(self.actions))

        # max steps
        self.max_steps = float('inf')

        # track the resolution for this maze
        self.resolution = 1

        # Rangel task modification
        self.TEMP_REWARD_STATES = [4,3]

    # take @action in @state
    # @return: [new state, reward]
    def step(self, state, action):
        x, y = state
        if action == self.ACTION_UP:
            x = max(x - 1, 0)
        elif action == self.ACTION_DOWN:
            x = min(x + 1, self.WORLD_HEIGHT - 1)
        elif action == self.ACTION_LEFT:
            y = max(y - 1, 0)
        elif action == self.ACTION_RIGHT:
            y = min(y + 1, self.WORLD_WIDTH - 1)
        if [x, y] in self.obstacles:
            x, y = state
        if [x, y] in self.GOAL_STATES:
            reward = 10
        # elif [x, y] in [[4,3]]: #self.TEMP_REWARD_STATES:
        #     reward = np.random.randn()
        #     self.TEMP_REWARD_STATES = []
        else:
            reward = np.random.randn() -1
        return [x, y], reward
    
    # Swith it up
    def switch(self):
        self.obstacles = [x for x in self.obstacles if x not in self.old_obstacles]
        self.GOAL_STATES = self.new_goal
        return


""" 
Modified mazes for experiments with Rangel.
"""
class RangelMaze:
    def __init__(self, version=0, stochastic_rewards=True):
        self.version = version
        self.stochastic_rewards = stochastic_rewards

        # maze width
        self.WORLD_WIDTH = 8

        # maze height
        self.WORLD_HEIGHT = 7

        # all possible actions
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_LEFT = 2
        self.ACTION_RIGHT = 3
        self.actions = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT]

        if self.version==0:
            # start state
            self.START_STATE = [4, 3]

            # goal state
            self.GOAL_STATES = [[0, 5]]

            # all obstacles
            self.obstacles = [[4, 0], [1, 3], [2, 3], [4, 5]]

            # positive reward states
            self.rewarding_states = [[2,1], [5,1], [2,6], [5,6]]

            # negative reward states
            self.aversive_states = [[2,4], [3,6]]
            
        elif self.version==1:
            self.START_STATE = [5, 2]

            # goal state
            self.GOAL_STATES = [[1, 5]]

            # all obstacles
            self.obstacles = [[2, 2], [3, 2], [4, 4], [4, 5]]

            # positive reward states
            self.rewarding_states = [[1,1], [5,6]]

            # negative reward states
            self.aversive_states = []

        # the size of q value
        self.q_size = (self.WORLD_HEIGHT, self.WORLD_WIDTH, len(self.actions))

        # max steps
        self.max_steps = float('inf')

        # track the resolution for this maze
        self.resolution = 1

    # take @action in @state
    # @return: [new state, reward]
    def step(self, state, action):
        x, y = state

        # next state
        if action == self.ACTION_UP:
            x = max(x - 1, 0)
        elif action == self.ACTION_DOWN:
            x = min(x + 1, self.WORLD_HEIGHT - 1)
        elif action == self.ACTION_LEFT:
            y = max(y - 1, 0)
        elif action == self.ACTION_RIGHT:
            y = min(y + 1, self.WORLD_WIDTH - 1)
        if [x, y] in self.obstacles:
            x, y = state

        # reward
        if [x, y] in self.GOAL_STATES:
            reward = 50
        elif [x, y] in self.rewarding_states:
            reward = 2
        elif [x, y] in self.aversive_states:
            reward = -3
        else:
            reward = -1
        if self.stochastic_rewards:
            reward += np.random.randn()

        return [x, y], reward


import random
import scipy.linalg
class BottleneckTask:
    # @transition_matrix:   State transition matrix
    # @reward_matrix:       Rewards corresponding to state transitions
    # @n_states:            Number of states
    # @states:              State indices
    def __init__(self, n_stages=2, stochastic_rewards=True,\
                stochastic_choice_sets=True):
        self.start_state = 0
        self.state = self.start_state
        self.stochastic_rewards = stochastic_rewards
        self.stochastic_choice_sets = stochastic_choice_sets

        # Defining one task stage
        self.module = np.mat('1 1 1 1 0 0 0 0 0 0 0;\
                              0 0 0 0 1 0 0 0 0 0 0;\
                              0 0 0 0 1 0 0 0 0 0 0;\
                              0 0 0 0 0 1 0 0 0 0 0;\
                              0 0 0 0 0 1 0 0 0 0 0;\
                              0 0 0 0 0 0 1 1 0 0 0;\
                              0 0 0 0 0 0 0 0 1 1 0;\
                              0 0 0 0 0 0 0 0 0 0 1;\
                              0 0 0 0 0 0 0 0 0 0 1;\
                              0 0 0 0 0 0 0 0 0 0 1;\
                              0 0 0 0 0 0 0 0 0 0 1')

        # Concatenating n_stages task stages
        trMat = np.kron(np.eye(n_stages), self.module)

        # Adding first and last states
        self.transition_matrix = np.zeros(\
                (trMat.shape[0]+1,trMat.shape[1]+1) )
        self.transition_matrix[:-1,1:] = trMat
        self.transition_matrix[-1,-1] = 1
        
        # Transitions, rewards, states
        self.n_states = len(self.transition_matrix)
        self.states = np.arange(self.n_states)
        self.transitions = np.transpose(self.transition_matrix.nonzero())
        
        # Defining rewards
        rewards1 = np.array([0, 140, 50, 100,  20, 0, 0, 20, -20, 20,  10])
        rewards2 = np.array([0,  60,  0,  20, -20, 0, 0, 20, -20, 20, -20])
        rewards3 = np.array([0, 140, 20, 100,  50, 0, 0, 20,  10, 20, -20])
        if n_stages == 2:
            self.rewards = np.hstack((rewards1, rewards2, 0))
        elif n_stages == 3:
            self.rewards = np.hstack((rewards1, rewards2, rewards3, 0))
        else:
            raise Exception('Current only supporting 2-3 stages')

        # actions available from each state
        self.actions = [list(self.transition_matrix[ii].nonzero()[0]) \
                        for ii in range(len(self.transition_matrix))]

        # the size of q value
        self.q_size = len(self.transitions)


    def reset(self):
        self.state = self.start_state
        return self.start_state

    # @property decorator 
    @property
    def action_space(self):
        assert self.state <= self.n_states
        if self.stochastic_choice_sets:
            if self.state not in [0,11]:
                return self.actions[self.state]

            elif self.state == 0:
                # Optimal agent visits states 5 or 6 with prob 0.5 & 0.5
                choiceList = [[1,2], [1,3], [1,4], [2,3], [2,4], [3,4]]
                return random.choices(choiceList, weights=(1,1,1,2,1,2), k=1)[0]

            elif self.state == 11:
                # Optimal agent visits states 16 or 17 with prob 0.8 and 0.2
                choiceList = [[12,13], [12,14], [12,15], [13,14], [13,15], [14,15]]
                return random.choices(choiceList, weights=(2,2,2,1,2,1), k=1)[0]

            elif self.state == 22:
                # Optimal agent visits states 16 or 17 with prob 0.8 and 0.2
                choiceList = [[23,24], [23,25], [23,26], [24,25], [24,26], [25,26]]
                return random.choices(choiceList, weights=(2,2,2,1,2,1), k=1)[0]
        else:
            return self.actions[self.state]

    # point to location in q-table
    def idx(self, state, action):
        ii  = np.logical_and(self.transitions[:,0]==state,\
                    self.transitions[:,1]==action)
        return int(np.argwhere(ii))

    # step in the environment
    def step(self, action):
        # next state
        next_state = action
        self.state = next_state

        # reward
        reward = self.rewards[next_state]
        if self.stochastic_rewards:
            reward += np.random.randn()

        # termination and info/logs
        done = False
        if next_state == self.n_states-1:
            done = True
        info = None

        return next_state, reward, done, info