import numpy as np
import random

class RangelTask:
    # @n_states:    Number of states
    # @states:      State indices
    # @pstate:      probability of state occurance
    # @task:        standard or precision measuring (PMT)
    def __init__(self, episodes_train=510, episodes_pmt=1020, n_pmt=20,
        learnPMT=False, delta_1=4, delta_2=1, delta_pmt=2):
        
        # states, acitons, and probability of occurance of states
        self.n_states = 12 * 3
        self.states = np.arange(self.n_states)
        #self.pstate = [.4]*6 + [.1]*6 + [0]*24
        
        # actions and size of the q-table
        self.actions = np.arange(12 * 3)    # normal, PMT+, PMT-
        self.q_size  = len(self.actions) + 1     # terminal

        # Defining rewards:
        self.delta_1 = delta_1
        self.delta_2 = delta_2
        self.rewards = [10 + self.delta_1,  10,  10 - self.delta_1,
                        10 + self.delta_2,  10,  10 - self.delta_2,
                        10 + self.delta_1,  10,  10 - self.delta_1,
                        10 + self.delta_2,  10,  10 - self.delta_2]
        # The steps below shouldn't be necessary, but I'm including them
        self.delta_pmt = delta_pmt
        self.rewards += list(np.array(self.rewards) + self.delta_pmt) +\
                        list(np.array(self.rewards) - self.delta_pmt)    

        
        # Experimental design: selecting the sequence of states
        self.episode = 0
        self.episodes_train = episodes_train
        self.episodes_pmt = episodes_pmt
        self.episodes = episodes_train + episodes_pmt
        self.learnPMT = learnPMT
        
        # pre-generating the sequence of states
        self.state_distribution = np.append(np.repeat(np.arange(6),4), np.arange(6,12), axis=0)
        
        self.states_training = np.repeat(self.state_distribution, self.episodes_train/len(self.state_distribution) )
        np.random.shuffle(self.states_training)
        np.random.shuffle(self.states_training)

        self.states_PMT = np.repeat(self.state_distribution, (self.episodes_pmt)/len(self.state_distribution) )
        np.random.shuffle(self.states_PMT)
        np.random.shuffle(self.states_PMT)

        self.states_pregen = np.append(self.states_training, self.states_PMT)
        # self.states_pregen = np.append(states_PMT[:(self.episodes-len(self.states_pregen))], self.states_pregen)
        
        # current state and next states
        self.state = self.states_pregen[self.episode]
        self.next_states = np.array([None]*len(self.states_pregen))
        
        # setting PMT trial indices and type (+Delta or -Delta)
        self.pmt_trial = np.zeros(len(self.states_pregen))
        self.n_pmt = n_pmt

        for ii in range(12):
            idx_ii = np.where(self.states_pregen == ii)[0]  # get index where state == ii
            idx_ii = idx_ii[idx_ii>self.episodes_train]     # throw away training indices
            np.random.shuffle(idx_ii)                       # shuffle what's left
            idx_i1 = idx_ii[:int(n_pmt/2)]                  # indices for +Delta PMT trials
            idx_i2 = idx_ii[int(n_pmt/2):n_pmt]             # indices for -Delta PMT trials

            # indicate pmt trial and type
            self.pmt_trial[idx_i1] =  1
            self.pmt_trial[idx_i2] = -1

            # for first half: next_state is option vs. +Delta deterministic option
            self.next_states[idx_i1] = self.states_pregen[idx_i1] + 12

            # for second half: -Delta deterministic option
            self.next_states[idx_i2] = self.states_pregen[idx_i2] + 24


    def reset(self, newEpisode=False):
        self.state = self.states_pregen[self.episode]
        if newEpisode:
            self.episode += 1
        if (self.episode == self.episodes):
            self.episode -= 1
            # print('Reached final episode.')
        return self.state


    @property
    def action_space(self):
        if self.state < 12:
            if self.state % 3 == 0:
                choiceSet = [self.state + 1, self.state + 2]    # 1 v 2; PMT 0
            elif self.state % 3 == 1:
                choiceSet = [self.state - 1, self.state + 1]    # 0 v 2; PMT 1
            else:
                choiceSet = [self.state - 2, self.state - 1]    # 0 v 1; PMT 2
        elif self.state < 24:
            choiceSet = [self.state - 12, self.state]
        elif self.state < 36:
            choiceSet = [self.state - 24, self.state]
        np.random.shuffle(choiceSet)
        return choiceSet

    # point to location in q-table
    def idx(self, state, action):
        idx = action
        return idx

    # step in the environment
    def step(self, action):
        # info: [updateQ?, allocGrad?]
        if self.state < 12:
            info = [True, True]
        else:
            info = [False, self.learnPMT]

        # next state
        if self.state < 12:
            next_state = self.next_states[self.episode]
            self.state = next_state
        else:
            next_state = None
        # self.next_states[self.episode] = None

        # reward
        reward = self.rewards[action]
        if action < 12:
            reward += np.random.randn()     # stochastic rewards

        # termination
        if next_state is None:  # if regular trial
           done = True
        else:
            done = False

        return next_state, reward, done, info




class BottleneckTask:
    # @transition_matrix:   State transition matrix
    # @reward_matrix:       Rewards corresponding to state transitions
    # @n_states:            Number of states
    # @states:              State indices
    def __init__(self, n_stages=2, stochastic_rewards=True,\
                stochastic_choice_sets=True, version=1):
        self.version = version
        self.start_state = 0
        self.state = self.start_state
        self.stochastic_rewards = stochastic_rewards
        self.stochastic_choice_sets = stochastic_choice_sets

        # Defining one stage of the task (to be repeated)
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
        
        # Defining rewards for version 1:
        # key states:                   if 3 stages:
        # p_visit = (.5, .5, .8, .2)    + (.8, .2)
        # dq      = (40, 20, 25, 40)    + (15, 30)
        rewards1 = np.array([0, 140, 50, 100,  20, 0, 0, 20, -20, 20,   0])
        rewards2 = np.array([0,  60,  0,  20, -20, 0, 0, 20,  -5, 20, -20])
        rewards3 = np.array([0, 140, 40, 100,  70, 0, 0, 20,   5, 20, -10])

        if version==2:
            # p_visit = (.8, .2, .8, .2)    + (.8, .2)
            # dq      = (40, 20, 25, 40)    + (15, 30)
            rewards3 = np.array([0, 140, 50,  100, 20, 0, 0, 20, 10, 20, -10])
        
        elif version == 3:
            # p_visit = (.6, .4, .8, .2)    + (.8, .2)
            # dq      = (40, 20, 25, 40)    + (15, 30)
            rewards1 = np.array([0, 140, 20, 100,  50, 0, 0, 20, -20, 20,   0])

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


    @property
    def action_space(self):
        assert self.state <= self.n_states
        
        if self.stochastic_choice_sets:

            if np.mod(self.state,11) != 0:
                # Bottleneck states are 0, 11, 22
                choiceSet = self.actions[self.state]

            elif self.state == 0:
                choiceList = [[1,2], [1,3], [1,4], [2,3], [2,4], [3,4]]

                if self.version == 1:
                    # p_visit(5,6) = (.5, .5)
                    choiceSet = random.choices(choiceList, weights=(1,1,1,2,1,2), k=1)[0]
                
                else:
                    # version 2: p_visit(5,6) = (.8, .2)
                    # version 3: p_visit(5,6) = (.6, .4)    (bec of diff. rewards)
                    choiceSet = random.choices(choiceList, weights=(2,2,2,1,2,1), k=1)[0]

            elif self.state == 11:
                # p_visit(16,17) = (.8, .2)
                choiceList = [[12,13], [12,14], [12,15], [13,14], [13,15], [14,15]]
                choiceSet = random.choices(choiceList, weights=(2,2,2,1,2,1), k=1)[0]

            elif self.state == 22:
                # p_visit(27,28) = (.8, .2)
                choiceList = [[23,24], [23,25], [23,26], [24,25], [24,26], [25,26]]
                choiceSet = random.choices(choiceList, weights=(2,2,2,1,2,1), k=1)[0]

        else:
            choiceSet = self.actions[self.state]
        
        return choiceSet

    # point to location in q-table
    def idx(self, state, action):
        ii  = np.logical_and(self.transitions[:,0]==state,\
                    self.transitions[:,1]==action)
        return np.searchsorted(ii, True) - 1

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



#########################################################################
#                                                                       #
# Other tasks that we tried running DRA on include the planning task    #
# from Huys et al 2015, a T-maze, a wrapper for arbitrary 2D gridwords  #
# which we used to create the Mattar & Daw 2019 maze, and a couple of   #
# mazes we tried for the human experiments with Antonio Rangel but      #
# eventually gave up on. Though they are not being used in the current  #
# experiments, they live below. In order to be used with the current    #
# DynamicResourceAllocator object, they need to be modified slightly    #
# to have the openAI gym-like structure as in the BottleneckTask above. #
#                                                                       #
#########################################################################

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