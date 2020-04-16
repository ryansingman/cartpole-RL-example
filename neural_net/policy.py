import torch
from torch import nn
from torch.autograd import Variable
from torch.distributions import Categorical
import numpy as np

class Policy(nn.Module):
    """Policy neural network module to enable policy gradient reinforcement learning"""
    def __init__(self, env, config):
        """
        Initializes policy neural net module
        Inputs;
            env -- learning environment
            config -- neural net config
        """
        super(Policy, self).__init__()

        # extract environment data
        self.env = env
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n

        self.config = config
        self.training_mode = True

        # initialize reward and policy history for episode
        self.policy_history = []
        self.reward_episode = []

        # initialize overall reward and policy history
        self.reward_history = []
        self.loss_history = []

        # generate model
        self.gen_model()

        # set epsilon
        self.episode_counter = 0
        self.max_episodes = self.config['max_episodes']
        self.update_epsilon()

    def gen_model(self):
        """
        Generates model based on config and training/testing
        Side Effects:
            self.model -- generates / updates model
        """
        # unpack config
        sizes = [self.state_space, *self.config['hidden_layer_sizes'], self.action_space]

        # initialize hidden layer(s) and output layer
        if self.training_mode:
            self.hls = [nn.Linear(l_size, sizes[n+1]) for n, l_size in enumerate(sizes[:-2])]
            self.relus = [nn.ReLU(inplace=True)] * len(self.hls) 
            self.ol = nn.Linear(sizes[-2], sizes[-1])
            dropouts = [nn.Dropout(self.config['dropout'])] * len(self.hls)

            layers = [layer for layers in zip(self.hls, dropouts, self.relus) for layer in layers]

        else:
            layers = [layer for layers in zip(self.hls, self.relus) for layer in layers]

        self.model = nn.Sequential(*layers, self.ol, nn.Softmax(dim=1))

    def forward(self, x):
        """
        Forward propagation
        Inputs:
            x -- input tensor
        Returns:
            y -- output tensor
        """
        return self.model(x)

    def action(self, x):
        """
        Outputs action based on policy, epsilon, and input tensor
        Inputs:
            x -- input tensor (representing environment state)
        Side Effects:
            
        Outputs:
            action -- action to take based on policy, epsilon, and state
        """
        x = torch.from_numpy(x).type(torch.FloatTensor).unsqueeze(0)
        probs = self.forward(x)
        m = Categorical(probs)
        if (np.random.uniform(0, 1)) < self.eps \
            and self.training_mode:
            action = np.random.choice(self.action_space)

        else:
            action = m.sample()

        # add policy choice to policy history
        action = torch.Tensor([action]).type(torch.FloatTensor)
        self.policy_history.append(m.log_prob(action))

        return action.type(torch.IntTensor)

    def update_reward(self, reward):
        """
        Appends latest reward to episode reward history
        Inputs:
            reward -- reward to append
        Side Effect:
            self.reward_episode -- appends list with latest reward
        """
        self.reward_episode.append(reward)

    def start_testing_mode(self):
        """
        Removes training layers from model for testing
        Side Effects:
            updates self.model
        """
        self.training_mode = False
        self.gen_model() 

    def update_epsilon(self):
        """
        Updates epsilon value using episode iteration
        Side Effects:
            self.eps -- updates epsilon
        """
        if self.episode_counter > 0:
            self.eps = np.clip(self.eps * (self.config['eps_rate']) ** (self.episode_counter / self.max_episodes), \
                               self.config['min_epsilon'], self.config['max_epsilon'])

        else:
            self.eps = self.config['max_epsilon']

        self.episode_counter += 1
