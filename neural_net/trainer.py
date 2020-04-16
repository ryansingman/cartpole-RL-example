import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import numpy as np

import pdb

class Trainer():
    """
    Trains neural net
    """
    def __init__(self, policy, optim_config):
        """
        Initializes trainer 
        Inputs:
            policy -- policy neural net module to optimize
            optim_config -- optimizer configuration
        """
        self.policy = policy

        self.conf = optim_config
        self.lr = optim_config['initial_lr']
        self.optim = optim.Adam(self.policy.parameters(), lr = self.lr)

        self.eps = np.finfo(np.float32).eps.item()

    def update_policy(self):
        """
        Trains model
        Inputs:
        """
        # get discounting factor and undiscounted rewards
        gamma = self.conf['gamma']
        undisc_rewards = self.policy.reward_episode

        # discount rewards
        rewards = [sum([reward * gamma**k for k, reward in enumerate(undisc_rewards[t:])]) for t in range(len(undisc_rewards))]
        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)

        # calculate policy gradient 
        policy_gradient = []
        for log_prob, reward in zip(self.policy.policy_history, rewards):
            policy_gradient.append(-log_prob * reward)

        # zero gradients
        self.optim.zero_grad()

        # perform backward pass and update weights and biases
        policy_gradient = torch.cat(policy_gradient).sum()
        policy_gradient.backward()
        self.optim.step()

        # update loss and reset policy history
        self.policy.loss_history.append(policy_gradient.item())
        self.policy.reward_history.append(np.sum(self.policy.reward_episode))
        self.policy.policy_history = []
        self.policy.reward_episode = []

        # update epsilon value
        self.policy.update_epsilon()

        # update learning rate
        self.update_lr()

        return policy_gradient

    def update_lr(self):
        """
        Updates learning rate
        Side Effects:
            self.lr -- decreased over time
        """
        if self.policy.episode_counter > 0:
            self.lr = np.clip(self.lr * (self.conf['lr_rate']) ** (self.policy.episode_counter / self.policy.max_episodes), \
                               self.conf['min_lr'], self.conf['initial_lr'])

        else:
            self.lr = self.conf['initial_lr']
