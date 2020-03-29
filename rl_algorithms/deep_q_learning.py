from collections import namedtuple
from itertools import count
import random
import matplotlib.pyplot as plt
import time
from utils import *

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
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, 30)
        self.fc3_bn = nn.BatchNorm1d(30)
        self.fc4 = nn.Linear(30, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3_bn(self.fc3(x)))
        return self.fc4(x)


class DeepQLearningAgent:

    def __init__(self, environment):
        self.environment = environment
        self.epsilon = 0.2
        self.batch_size = 64
        self.gamma = 0.9
        self.target_update = 10
        self.memory = ReplayMemory(1000000)

        self.state_space_dims = environment.state_space_dims
        self.n_actions = environment.n_actions
        self.actions = environment.action_space.values

        self.policy_net = DQN(self.state_space_dims, self.n_actions)
        self.target_net = DQN(self.state_space_dims, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.policy_net.train() #train mode (train vs eval mode)

        self.optimizer = optim.Adam(self.policy_net.parameters()) #todo pokusaj nesto drugo
        #self.optimizer = optim.RMSprop(self.policy_net.parameters())

    #return set point on the energy storage
    def get_action(self, state, epsilon):
        if random.random() > epsilon:
            self.policy_net.eval()
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action_index = self.policy_net(state).max(1)[1].view(1, 1)
                self.policy_net.train()
                #return torch.tensor([[self.actions[action_index]]])
                return self.actions[action_index]

        else:
            #return torch.tensor([[self.actions[random.randint(0, len(self.actions)-1)]]], dtype=torch.float)   
            return self.actions[random.randint(0, len(self.actions)-1)] 

    def environment_reset(self, df_day):
        first_row = df_day.iloc[0]
        solar_percents, load_percents = get_scaling_from_row(first_row)
        return self.environment.reset(solar_percents, load_percents)


    def train(self, df_train, n_episodes):
        #self.policy_net.load_state_dict(torch.load("policy_net"))
        total_episode_rewards = []
        for i_episode in range(n_episodes):
            if (i_episode % 1 == 0):
                print("==============================Episode: ", i_episode)
            #if (i_episode == int(0.05 * n_episodes)):
                #self.epsilon = 0.1

            done = False

            df_train_day = select_random_day(df_train)
            state = self.environment_reset(df_train_day)
            state = torch.tensor([state], dtype=torch.float)
            total_episode_reward = 0

            #prvi red (i = 0) je gore sluzio za inicijalno stanje
            #trenutni red se koristi da se dobave scaling vrijednosti za sljedeci timestep
            #za i = len(df_train_day) ce next_state biti None, done = True, ali i tada hocemo da odradimo environment.step
            for next_timestep_idx in range(1, len(df_train_day)+1):
                action = self.get_action(state, self.epsilon)
                if abs(action) > 1.0:
                    print('Warning: deep_q_learning.train - abs(action) > 1')
                if (next_timestep_idx < len(df_train_day)):
                    row = df_train_day.iloc[next_timestep_idx]
                    next_solar_percents, next_load_percents = get_scaling_from_row(row)
                #else neka percents zadrze stare vrijednosti, necemo ih ni koristiti

                next_state, reward, done = self.environment.step(action = action, solar_percents = next_solar_percents, load_percents = next_load_percents)
                total_episode_reward += reward
                reward = torch.tensor([reward], dtype=torch.float)
                action = torch.tensor([action], dtype=torch.float)
                next_state = torch.tensor([next_state], dtype=torch.float)
                if done:
                    next_state = None

                self.memory.push(state, action, next_state, reward)
                state = next_state
                self.optimize_model()

                #dodaj provjeru da li je self.environment.timestamp != index%24 + 1
                if (next_timestep_idx != self.environment.timestep):
                    print('Warning: deep_q_learning.train - something may be wrong with timestep indexing')

            if (i_episode % 50 == 0):
                print ("total_episode_reward: ", total_episode_reward)

            total_episode_rewards.append(total_episode_reward)
            
            if (i_episode % 200 == 199):
                torch.save(self.policy_net.state_dict(), "policy_net")
                time.sleep(3)

            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        torch.save(self.policy_net.state_dict(), "policy_net")

        x_axis = [1 + j for j in range(len(total_episode_rewards))]
        plt.plot(x_axis, total_episode_rewards)
        plt.xlabel('Episode number') 
        plt.ylabel('Total episode reward') 
        plt.savefig("total_episode_rewards.png")
        plt.show()


    def test(self, df_test):
        print('agent testing started')
        self.policy_net.load_state_dict(torch.load("policy_net"))
        self.policy_net.eval()

        day_starts = extract_day_starts(df_test)
        day_start_times = list(day_starts.time)
        for day_start_time in day_start_times: 
            df_test_day = df_test[(df_test.time >= day_start_time) & (df_test.time < day_start_time + 24)]

            done = False
            state = self.environment_reset(df_test_day)
            state = torch.tensor([state], dtype=torch.float)
            total_episode_reward = 0
            #todo neka ove promjenljive budu ukupne snage u mrezi u aps. jedinicama
            solar_powers = []
            load_powers = []
            storage_powers = []

            #inicijalni red iz dataframe sluzi za inicijalizaciju, on se u okviru predstojece petlje preskace
            first_row = df_test_day.iloc[0]
            first_solar_percents, first_load_percents = get_scaling_from_row(first_row)
            solar_powers.append(first_solar_percents[0] * -1)
            load_powers.append(first_load_percents[0])

            for next_timestep_idx in range(1, len(df_test_day)+1):
                action = self.get_action(state, epsilon = 0.0)
                storage_powers.append(action)
                if abs(action) > 1.0:
                    print('Warning: deep_q_learning.train - abs(action) > 1')
                if (next_timestep_idx < len(df_test_day)):
                    row = df_test_day.iloc[next_timestep_idx]
                    next_solar_percents, next_load_percents = get_scaling_from_row(row)
                    solar_powers.append(next_solar_percents[0] * -1)
                    load_powers.append(next_load_percents[0])

                next_state, reward, done = self.environment.step(action = action, solar_percents = next_solar_percents, load_percents = next_load_percents)
                total_episode_reward += reward
                print('action', action)
                print(state)
                state = torch.tensor([next_state], dtype=torch.float)
            print('total_episode_reward', total_episode_reward)
            plot_daily_results(int(day_start_time/24 + 1), solar_powers, load_powers, storage_powers)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)

        #converts batch array of transitions to transiton of batch arrays
        batch = Transition(*zip(*transitions))

        #compute a mask of non final states and concatenate the batch elements
        #there will be zero q values for final states later... therefore we need mask
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype = torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).view(-1,1)
        reward_batch = torch.cat(batch.reward).view(-1,1)

        # compute Q(s_t, a) - the model computes Q(s_t), then we select
        # the columns of actions taken. These are the actions which would've
        # been taken for each batch state according to policy net
        action_indices = (action_batch / self.environment.action_space.step).round() + self.environment.action_space.size // 2
        action_indices = action_indices.to(dtype=torch.int64)
        state_action_values = self.policy_net(state_batch).gather(1, action_indices)

        #gather radi isto sto i:
        #q_vals = []
        #for qv, ac in zip(Q(obs_batch), act_batch):
        #    q_vals.append(qv[ac])
        #q_vals = torch.cat(q_vals, dim=0)

        # Compute V(s_{t+1}) for all next states
        # q values of actions for non terminal states are computed using
        # the older target network, selecting the best reward with max
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach() #manje od 128 stanja, nema final stanja
        #za stanja koja su final ce next_state_values biti 0
        #detach znaci da se nad varijablom next_state_values ne vrsi optimizacicja
        next_state_values = next_state_values.view(-1,1)
        # compute the expected Q values
        expected_state_action_values = (next_state_values*self.gamma) + reward_batch

        #Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        #todo razmisli
        #for param in self.policy_net.parameters():
            #param.grad.data.clamp_(-1, 1)
        self.optimizer.step()