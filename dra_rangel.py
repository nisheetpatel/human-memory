import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from task import RangelTask

class DynamicResourceAllocator:
    def __init__(self, model='dra', lmda=0.1, sigmaBase=5,\
                learnPMT=False, delta_pmt=2, delta_1=4, delta_2=1,\
                learning_sigma=0.2, learning_q=0.2, discount=0.95,\
                nTraj=10, beta=10, gradient='A', nGradUpdates=5, \
                updateFreq=25, decay=1, decay1=0.98,\
                printFreq=50, printUpdates=True):

        self.env            = RangelTask(learnPMT=learnPMT, \
            delta_pmt=delta_pmt, delta_1=delta_1, delta_2=delta_2)
        self.learning_q     = learning_q
        self.learning_sigma = learning_sigma
        self.discount       = discount
        self.gradient       = gradient
        self.nTraj          = nTraj
        self.lmda           = lmda
        self.beta           = beta
        self.printFreq      = printFreq
        self.printUpdates   = printUpdates
        self.updateFreq     = updateFreq
        self.nGradUpdates   = nGradUpdates
        self.model          = model
        
        models = ['dra','equalPrecision','freqBased']
        assert model in models, f"Invalid model type. \
                    \n Model must be one of {models}."

        # Initialising q-distribution: N(q, diag(sigma)^2)
        self.q  = np.zeros(self.env.q_size)
        self.sigma0     = sigmaBase/2
        self.sigmaBase  = sigmaBase
        self.sigma      = np.ones(self.q.shape) * self.sigma0

        # Indices to fix: sigma = 0 and q = rewards (known)
        self.fixed_ids = np.arange(12,len(self.sigma)-1)
        self.q[self.fixed_ids] = np.array(self.env.rewards)[self.fixed_ids]
        self.sigma[self.fixed_ids] = 0
        self.sigma[-1] = 0

        # Initialising visit frequency
        self.n_visits   = 1e-5*np.ones(self.sigma.shape)
        self.n_visits_w = 1e-5*np.ones(self.sigma.shape)
        self.n_visits_w1= 1e-5*np.ones(self.sigma.shape)
        self.decay      = decay
        self.decay1     = decay1
        self.c          = 0.1     # could be optimized

        # Initialising array to record choices
        self.choicePMT  = np.nan * np.ones(self.env.episodes)


    def softmax(self, x):
        x = np.array(x)
        b = np.max(self.beta*x)      # Trick to avoid overflow errors
        y = np.exp(self.beta*x - b)  # during numerical calculations
        return y/y.sum()


    def act(self, state):
        """ 
        Soft Thompson sampling policy: takes one sample from each of
        the memories corresponding the the choices available in the 
        state and returns an action. Also returns the sample drawn 
        (zeta), probability of taking action acc. to softmax, and 
        the choice set available. These are required for resource
        allocation by SGD for DRA.
        """
        # fetch available actions and pointer for q-table
        actions = self.env.action_space
        idx     = [self.env.idx(state,a) for a in actions]

        # random draws from memory distribution
        zeta_s  = np.random.randn(len(actions))
        prob_a  = self.softmax(self.q[idx] + zeta_s*self.sigma[idx])

        # chosen action
        a       = np.random.choice(np.arange(len(prob_a)), p=prob_a)
        action  = actions[a]

        return action, zeta_s, prob_a, actions

    
    def runEpisode(self, updateQ=True, newEp=False):
        """
        Run one episode of the task, update the memorized q-values if
        updateQ is True, and return reward obtained in the episode.
        """
        # Initializing some variables
        tot_reward, reward = 0, 0
        done = False
        state = self.env.reset(newEpisode=newEp)
        idx_list = []

        while not done:
            # Determine next action
            action, _, _, _ = self.act(state)

            # record choices for PMT trials:
            if self.env.pmt_trial[self.env.episode]:
                    self.choicePMT[self.env.episode] = action

            # Get next state and reward
            s1, reward, done, info = self.env.step(action)
            updateQ = updateQ * info[0]
            allocGrad = info[1]

            # find pointers to q-table
            idx   = self.env.idx(state,action)
            idx_list.append(idx)
            if not done:
                a1, _, _, _ = self.act(s1)    # next action for sarsa
                idx1  = self.env.idx(s1,a1)
            else:
                idx1  = -1

            if updateQ:
                # SARSA update for q-value
                if idx < 12:    # only update q-values for first 12 states
                    delta = reward - self.q[idx]    # target is just reward
                    self.q[idx] += self.learning_q * delta

            # Update state and total reward obtained
            state = s1
            tot_reward += reward

            # Update visits for each idx during episode
            # done for current idx IFF newEp and allocGrad
            if (newEp & allocGrad):
                self.n_visits[idx] += 1
                self.n_visits_w = self.decay * self.n_visits_w
                self.n_visits_w[idx] += 1
                self.n_visits_w1= self.decay1 * self.n_visits_w1
                self.n_visits_w1[idx] += 1
        
        # reset q-values for terminal state (unnecessary)
        # q[fixed_ids] should never be updated because updateQ=False
        self.q[-1] = 0      # terminal state

        return tot_reward


    def computeExpectedReward(self):
        """
        Compute expected reward for current memory distribution.
        """
        rewards = []
        env_currentEp = self.env.episode
        for ep in range(self.env.episodes_train):
            self.env.episode = ep
            rewards.append(self.runEpisode(updateQ=False, newEp=False))
        self.env.episode = env_currentEp
        return np.mean(rewards)


    @staticmethod                    # cost function
    def kl_mvn(m0, S0, m1, S1):
        """
        KL-divergence from Gaussian m0,S0 to Gaussian m1,S1,
        expressed in nats. Diagonal covariances are assumed.
        """    
        # store inv diag covariance of S1 and diff between means
        N = m0.shape[0]
        iS1 = np.linalg.inv(S1)
        diff = m1 - m0

        # three terms of the KL divergence
        tr_term   = np.trace(iS1 @ S0)
        # det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0))
        det_term  = np.trace(np.ma.log(S1)) - \
                    np.trace(np.ma.log(S0))
        quad_term = diff.T @ np.linalg.inv(S1) @ diff 
        return .5 * (tr_term + det_term + quad_term - N)


    def computeCost(self):
        """
        Computes the cost of representing memories precisely, defined
        as the KL-divergence from the base distribution.
        """
        q = self.q.flatten()
        S1 = np.diag(np.square(self.sigma.flatten()))
        S0 = np.diag(np.square(np.ones(len(self.sigma.flatten()))\
            *self.sigmaBase))
        return (self.lmda * self.kl_mvn(q,S1,q,S0))

    
    def minusObjective(self, sigma_scalar):
        """
        CAUTION: Use ONLY for gradient-free optimization of equal 
        precision and freq. based models. Takes in sigma_scalar and 
        outputs noisy objective as a function of sigma. In the 
        process, it ends up resetting self.sigma.
        """
        if self.model == 'equalPrecision':
            self.sigma[:] = sigma_scalar
        
        elif self.model == 'freqBased':
            self.sigma = sigma_scalar / \
                (self.c + np.sqrt(self.n_visits_w))

        # Set terminal states' and fixed_ids sigma to sigmaBase
        # these are known with certainty and shouldn't cost anything
        self.sigma[-1] = self.sigmaBase
        self.sigma[self.fixed_ids] = self.sigmaBase

        # Set all sigmas greater than sigmaBase to sigmaBase
        self.sigma[self.sigma > self.sigmaBase] = self.sigmaBase

        # Objective function to return
        minusObj = self.computeCost() - self.computeExpectedReward()

        # Resetting sigma for terminal states and fixed ids to 0
        self.sigma[-1] = 0
        self.sigma[self.fixed_ids] = 0

        return minusObj


    def allocateResources(self):
        """
        Allocate resources depending on the model:
        if self.model == 'dra':
            The precision of each memory gets updated independently
            acc. to the gradient in Patel et al. NeurIPS 2020.

        elif self.model == 'equalPrecision':
            All memories are constrained to be equally precise. The
            common precision of all memories is updated via black-box
            (or gradient-free) optimization.

        elif self.model == 'freqBased':
            The precision of memories is proportional to the number
            of visits to the corresponding state-action pair. The 
            proportionality constant is updated via black-box
            optimization, as in the equalPrecision model.
        """
        # Gradient-based optimization for DRA
        if self.model == 'dra':
            grads = []

            for _ in range(int(self.nTraj)):
                # Initialising some variables
                grad  = np.zeros(self.sigma.shape)
                done = False
                tot_reward, reward = 0, 0
                state = self.env.reset()  # newEp=False by default
                r = []

                while not done:
                    # Determine next action
                    action, z, prob_a, action_space = self.act(state)
                    
                    # Get next state and reward
                    s1, reward, done, info = self.env.step(action)
                    allocGrad = info[1]
                    r.append(reward)

                    if allocGrad:
                        # find pointers to q-table
                        idx    = self.env.idx(state,action)
                        id_all = [self.env.idx(state,a) \
                                    for a in action_space]

                        if self.gradient == 'A':
                            psi = self.q[idx] - np.mean(self.q[id_all])
                        elif self.gradient == 'Q':
                            psi = self.q[idx]
                        elif self.gradient == 'R':
                            psi = 1
                    
                        # gradients
                        grad[id_all] -= (self.beta * z * prob_a) * psi
                        grad[idx]    += psi * (self.beta * \
                                z[np.array(action_space) == action])

                    # Update state for next step, add total reward
                    state       = s1
                    tot_reward += reward
                
                if self.gradient == 'R':
                    rturn = np.sum(r)
                    grads += [np.dot(rturn,grad)]
                else:
                    # Collect sampled stoch. gradients for all trajs
                    grads   += [grad]
                    # reward_list.append(tot_reward)

            # Setting fixed and terminal sigmas to sigmaBase to avoid
            # divide by zero error; reset to 0 at the end of the loop
            self.sigma[self.fixed_ids] = self.sigmaBase
            self.sigma[-1] = self.sigmaBase
            
            # Compute average gradient across sampled trajs & cost
            grad_cost = (self.sigma/(self.sigmaBase**2) - 1/self.sigma)
            grad_mean = np.mean(grads, axis=0)

            # Updating sigmas
            self.sigma += self.learning_sigma * \
                            (grad_mean - self.lmda * grad_cost)


        # Gradient-free optimization
        else:
            # set upper bound for search and normalization
            if self.model == 'freqBased':
                norm = (self.c + np.sqrt(self.n_visits_w))
                ub = np.max(norm) * self.sigmaBase

            elif self.model == 'equalPrecision':
                norm = np.ones(self.sigma.shape)
                ub = self.sigmaBase

            # Optimize
            res = minimize_scalar(self.minusObjective,\
                    method='Bounded', bounds=[0.01,ub])
            self.sigma = res.x / norm
        
        # For all models
        # Clip sigma to be less than sigmaBase
        self.sigma = np.clip(self.sigma, 0.01, self.sigmaBase)

        # Sigmas for fixed ids and terminal states = 0
        # works because cost function altered to ignore this
        self.sigma[self.fixed_ids] = 0
        self.sigma[-1] = 0       



    def train(self):
        # Initialize variables to track rewards
        reward_list = []
        ave_reward_list = []
        
        for ii in range(self.env.episodes-1):
            # run episode
            rewardObtained = self.runEpisode(newEp=True)

            # printing updates to indicate training progress
            reward_list.append(rewardObtained)
            if ((ii+1) % self.printFreq == 0) & self.printUpdates:
                ave_reward = np.mean(reward_list)
                ave_reward_list.append(ave_reward)
                reward_list = []
                print(f'Episode {ii+1}, \
                    Objective = {np.around(ave_reward-self.computeCost(),2)}')
            
            # resource allocation:
            if self.model == 'dra':
                # 10 gradient updates per episode for DRA
                for _ in range(self.nGradUpdates):
                    self.allocateResources()
            else:
                # find optimum every updateFreq episodes
                if (ii+1) % self.updateFreq == 0:
                    self.allocateResources()

        return


    @property
    def memoryTable(self):
        df = pd.DataFrame( {
            'state':        np.repeat(['s1','s2','s3','s4'],3),\
            'action':       self.env.actions[:12], \
            'q':            np.around(self.q[:12],1),\
            'sigma':        np.around(self.sigma[:12],1) })
        return df


# Saving stuff
if __name__ == '__main__':
    # basic script
    from dra_rangel import DynamicResourceAllocator
    import pandas as pd
    import numpy as np
    import pickle

    model = DynamicResourceAllocator(learnPMT=False)
    model.train()
    df = model.memoryTable
    print(df)
    df.to_pickle(f'./figures/df_{model.model}')