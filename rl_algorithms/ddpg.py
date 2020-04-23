import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd
from torch.autograd import Variable

import numpy as np
from collections import deque
import random
import time
import matplotlib.pyplot as plt
from utils import *

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)

        init_w = 3e-3
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = self.linear4(x) #returns q value, should not be limited by tanh
        return x

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate = 3e-4):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)

        init_w = 3e-3
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = torch.relu(self.linear1(state))
        x = torch.relu(self.linear2(x))   
        x = torch.relu(self.linear3(x))
        x = torch.tanh(self.linear4(x))
        return x

class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.1, max_sigma=0.7, min_sigma=0.7, decay_period=100):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high

        self.reset()        

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t = 0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma-self.min_sigma) * min(1.0, t/self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)

#scales action from [-1, 1] to [action_space.low, action_space.high]
def reverse_action(action, action_space):
    act_k = (action_space.high - action_space.low)/ 2.
    act_b = (action_space.high + action_space.low)/ 2.
    return act_k * action + act_b  

#scales action from [action_space.low, action_space.high] to [-1, 1]
def normalize_action(action, action_space):
    act_k_inv = 2./(action_space.high - action_space.low)
    act_b = (action_space.high + action_space.low)/ 2.
    return act_k_inv * (action - act_b)

#scales action tensor from [-1, 1] to [action_space.low, action_space.high]
def reverse_action_tensor(action, action_space):
    high = np.asscalar(action_space.high)
    low = np.asscalar(action_space.low)
    act_k = (high - low)/ 2.
    act_b = (high + low)/ 2.
    act_b_tensor = act_b * torch.ones(action.shape)
    return act_k * action + act_b_tensor

class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)

class DDPGAgent:
    def __init__(self, environment, hidden_size=128, actor_learning_rate=1e-5, critic_learning_rate=1e-4, gamma=1.0, tau=1e-3, max_memory_size=600000):
        self.environment = environment
        self.num_states = environment.state_space_dims
        self.num_actions = environment.action_space_dims
        self.gamma = gamma
        self.tau = tau
        self.timestep = 0

        self.batch_size = 256

        self.actor = Actor(self.num_states, hidden_size, self.num_actions)
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions)
        self.actor.train()
        self.critic = Critic(self.num_states+self.num_actions, hidden_size, self.num_actions)
        self.critic_target = Critic(self.num_states+self.num_actions, hidden_size, self.num_actions)
        self.critic.train()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            
        self.replay_buffer = ReplayBuffer(max_memory_size)
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        
        self.noise = OUNoise(self.environment.action_space)

    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        self.actor.eval()
        action = self.actor.forward(state)
        self.actor.train()
        action = action.detach().cpu().numpy()[0,0]
        #output from actor network is normalized so:
        action = reverse_action(action, self.environment.action_space)
        return action

    def update(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        #print('states', states)
        #print('actions', actions)

        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        #output from actor network is normalized so:
        next_actions = reverse_action_tensor(next_actions, self.environment.action_space)
        
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + (1.0 - dones) * self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime.detach())

        #actor loss
        next_actions_pol_loss = self.actor.forward(states)
        next_actions_pol_loss = reverse_action_tensor(next_actions_pol_loss, self.environment.action_space)
        policy_loss = -self.critic.forward(states, next_actions_pol_loss).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data*self.tau + target_param.data*(1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data*self.tau + target_param.data*(1.0 - self.tau))

    def environment_reset(self, df_day):
        first_row = df_day.iloc[0]
        solar_percents, load_percents, electricity_price = get_scaling_from_row(first_row)
        return self.environment.reset(solar_percents, load_percents, electricity_price)

    
    def train(self, df_train, n_episodes):
        #self.actor.load_state_dict(torch.load("model_actor"))
        #self.critic.load_state_dict(torch.load("model_critic"))
        total_episode_rewards = []
        self.moving_average = 0.0
        for i_episode in range(n_episodes):
            if (i_episode % 20 == 0):
                print("Episode: ", i_episode)
                
            if (i_episode == 10000):
                self.noise.min_sigma = 0.3
                self.noise.max_sigma = 0.3

            df_train_day = select_random_day(df_train)
            state = self.environment_reset(df_train_day)

            self.noise.reset()
            done = False
            episode_iterator = 0
            total_episode_reward = 0 
            self.timestep = 0

            #prvi red (i = 0) je gore sluzio za inicijalno stanje
            #trenutni red se koristi da se dobave scaling vrijednosti za sljedeci timestep
            #za i = len(df_train_day) ce next_state biti None, done = True, ali i tada hocemo da odradimo environment.step
            for next_timestep_idx in range(1, len(df_train_day)+1):
                state = np.asarray(state)
                action = self.get_action(state)
                action = self.noise.get_action(action, self.timestep)
                if abs(action) > 1.0:
                    print('Warning: deep_q_learning.train - abs(action) > 1')

                if (next_timestep_idx < len(df_train_day)):
                    row = df_train_day.iloc[next_timestep_idx]
                    next_solar_percents, next_load_percents, electricity_price = get_scaling_from_row(row)

                next_state, reward, done, _, _ = self.environment.step(action = action, solar_percents = next_solar_percents, load_percents = next_load_percents, electricity_price = electricity_price)
                total_episode_reward += reward
                self.timestep += 1
                
                self.replay_buffer.push(state, action, reward, next_state, done)
                if len(self.replay_buffer) > self.batch_size:
                    self.update()

                state = next_state

                if (next_timestep_idx != self.environment.timestep):
                    print('Warning: deep_q_learning.train - something may be wrong with timestep indexing')

            if (i_episode == 0):
                self.moving_average = total_episode_reward
            self.moving_average = 0.9*self.moving_average + 0.1*total_episode_reward
            total_episode_rewards.append(self.moving_average)
            
            if (i_episode % 20 == 0):
                print ("total_episode_reward: ", total_episode_reward)
            
            if (i_episode % 1000 == 999):
                time.sleep(60)

            if (i_episode % 20 == 0):
                torch.save(self.actor.state_dict(), "./trained_nets/model_actor"+str(i_episode))
                #torch.save(self.critic.state_dict(), "./trained_nets/model_critic"+str(i_episode))                
        
        torch.save(self.actor.state_dict(), "model_actor")
        torch.save(self.critic.state_dict(), "model_critic")
        
        x_axis = [1 + j for j in range(len(total_episode_rewards))]
        plt.plot(x_axis, total_episode_rewards)
        plt.xlabel('Episode number') 
        plt.ylabel('Total episode reward') 
        plt.savefig("total_episode_rewards.png")
        plt.show()

    def test(self, df_test):
        print('agent testing started')
        self.actor.load_state_dict(torch.load("model_actor"))
        self.actor.eval()

        day_starts = extract_day_starts(df_test)
        day_start_times = list(day_starts.time)
        for day_start_time in day_start_times: 
            df_test_day = df_test[(df_test.time >= day_start_time) & (df_test.time < day_start_time + 24)]
            
            done = False
            state = self.environment_reset(df_test_day)
            total_episode_reward = 0
            #todo neka ove promjenljive budu ukupne snage u mrezi u aps. jedinicama
            solar_powers = []
            load_powers = []
            proposed_storage_powers = []
            actual_storage_powers = []
            storage_socs = []
            electricity_price = []

            #inicijalni red iz dataframe sluzi za inicijalizaciju, on se u okviru predstojece petlje preskace
            first_row = df_test_day.iloc[0]
            first_solar_percents, first_load_percents, first_electricity_price = get_scaling_from_row(first_row)
            solar_powers.append(first_solar_percents[0] * -1)
            load_powers.append(first_load_percents[0])
            electricity_price.append(first_electricity_price[0])

            for next_timestep_idx in range(1, len(df_test_day)+1):
                state = np.asarray(state)
                action = self.get_action(state)
                proposed_storage_powers.append(action)
                if abs(action) > 1.0:
                    print('Warning: deep_q_learning.train - abs(action) > 1')
                if (next_timestep_idx < len(df_test_day)):
                    row = df_test_day.iloc[next_timestep_idx]
                    next_solar_percents, next_load_percents, next_electricity_price = get_scaling_from_row(row)
                    solar_powers.append(next_solar_percents[0] * -1)
                    load_powers.append(next_load_percents[0])
                    electricity_price.append(next_electricity_price[0])

                next_state, reward, done, actual_action, initial_soc = self.environment.step(action = action, solar_percents = next_solar_percents, load_percents = next_load_percents, electricity_price = next_electricity_price)
                
                actual_storage_powers.append(actual_action)
                storage_socs.append(initial_soc)

                total_episode_reward += reward
                state = next_state
        print('total_episode_reward', total_episode_reward)
        plot_daily_results(int(day_start_time/24 + 1), solar_powers, load_powers, proposed_storage_powers, actual_storage_powers, storage_socs, electricity_price)