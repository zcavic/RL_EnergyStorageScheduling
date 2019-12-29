from collections import namedtuple
from itertools import count
import random
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc3_bn = nn.BatchNorm1d(50)
        self.fc4 = nn.Linear(50, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3_bn(self.fc3(x)))
        return self.fc4(x)


class DeepQLearningAgent:

    def __init__(self, environment):
        self.environment = environment
        self.epsilon = 0.5
        self.batch_size = 32
        self.gamma = 1.0
        self.target_update = 5
        self.memory = ReplayMemory(1000000)

        self.state_space_dims = environment.state_space_dims
        self.n_actions = environment.n_actions

        self.policy_net = DQN(self.state_space_dims, self.n_actions)
        self.target_net = DQN(self.state_space_dims, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.policy_net.train() #train mode (train vs eval mode)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.00001) #todo pokusaj nesto drugo
        #self.optimizer = optim.RMSprop(self.policy_net.parameters())

    #return set point on the energy storage
    def get_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action_index = self.policy_net(state).max(1)[1].view(1, 1)
                return self.get_action_by_index(action_index)
        else:
            return torch.tensor([[self.actions[random.randint(0, len(self.actions)-1)]]], dtype=torch.float)     

    def train(self, df_train, n_episodes):
        #akcije neka budu slicne kao u HVAC projektu
        #iscrtavanje gubitaka po epizodi i citanje redova iz dataframe-a kao u VVO projektu
        pass


    def test(self, df_test):
        pass

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)

        #converts batch array of transitions to transiton of batch arrays
        batch = Transition(*zip(*transitions))

        #compute a mask of non final states and concatenate the batch elements
        #final states will have zero q values
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype = torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).view(-1,1)
        reward_batch = torch.cat(batch.reward).view(-1,1)

        # compute Q(s_t, a) - the model computes Q(s_t), then we select
        # the columns of actions taken. These are the actions which would've
        # been taken for each batch state according to policy net
        action_indices = (action_batch / self.environment.action_space.step).round() #todo transfer this in some method in Incremental space
        action_indices = action_indices.to(dtype=torch.int64)
        state_action_values = self.policy_net(state_batch).gather(1, action_indices)

        # Compute V(s_{t+1}) for all next states
        # q values of actions for non terminal states are computed using
        # the older target network, selecting the best reward with max
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        #for final states next_state_values will be 0
        #detach - variable isn't being optimizes
        next_state_values = next_state_values.view(-1,1)
        # compute the expected Q values
        expected_state_action_values = (next_state_values*self.gamma) + reward_batch

        #Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()