import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

class SACNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, alpha, beta, max_action,
                 fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/sac'):
        super(SACNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.chkpt_dir = chkpt_dir
        if not os.path.exists(self.chkpt_dir):
            os.makedirs(self.chkpt_dir)

        self.critic = CriticNetwork(beta, input_dims, n_actions, fc1_dims, fc2_dims)
        self.value = ValueNetwork(beta, input_dims, fc1_dims, fc2_dims)
        self.actor = ActorNetwork(alpha, input_dims, max_action, fc1_dims, fc2_dims, n_actions)

    def forward(self, state, action):
        value = self.value(state)
        mu, sigma = self.actor.forward(state)
        critic_value = self.critic.forward(state, action)

        return value, mu, sigma, critic_value

    def save_models(self):
        self.critic.save_checkpoint()
        self.value.save_checkpoint()
        self.actor.save_checkpoint()

    def load_models(self):
        self.critic.load_checkpoint()
        self.value.load_checkpoint()
        self.actor.load_checkpoint()

class CriticNetwork(nn.Module):


class ValueNetwork(nn.Module):


class ActorNetwork(nn.Module):
