import torch
import torch.autograd
import numpy as np


class Action:
    def __init__(self, timestamp, value):
        self.timestamp = timestamp
        self.value = value


# scales action from [-1, 1] to [action_space.low, action_space.high]
def reverse_action(action, action_space):
    act_k = (action_space.high - action_space.low) / 2.
    act_b = (action_space.high + action_space.low) / 2.
    return act_k * action + act_b


# scales action from [action_space.low, action_space.high] to [-1, 1]
def normalize_action(action, action_space):
    act_k_inv = 2. / (action_space.high - action_space.low)
    act_b = (action_space.high + action_space.low) / 2.
    return act_k_inv * (action - act_b)


# scales action tensor from [-1, 1] to [action_space.low, action_space.high]
def reverse_action_tensor(action, action_space):
    high = np.asscalar(action_space.high)
    low = np.asscalar(action_space.low)
    act_k = (high - low) / 2.
    act_b = (high + low) / 2.
    act_b_tensor = act_b * torch.ones(action.shape)
    return act_k * action + act_b_tensor