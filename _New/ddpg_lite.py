import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd
import numpy as np
from utils import *
from torch.autograd import Variable
from _New.action import reverse_action_tensor, reverse_action
from _New.actor import Actor
from _New.critic import Critic
from _New.ou_noise import OUNoise
from _New.replay_buffer import ReplayBuffer


class DDPGAgentLite:
    def __init__(self, environment, hidden_size=128, actor_learning_rate=1e-5, critic_learning_rate=1e-4, gamma=1.0,
                 tau=1e-3, max_memory_size=600000):
        self.environment = environment
        self.num_states = environment.state_space_dims
        self.num_actions = environment.action_space_dims
        self.gamma = gamma
        self.tau = tau
        self.timestamp = 0
        self.moving_average = 0
        self.batch_size = 256
        self.actor = Actor(self.num_states, hidden_size, self.num_actions)
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions)
        self.actor.train()
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)
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

    def train(self, n_episodes):
        total_episode_rewards = []
        self.moving_average = 0.0
        for i_episode in range(n_episodes):
            if i_episode % 50 == 0:
                print("Episode: ", i_episode)
            if i_episode == 10000:
                self.noise.min_sigma = 0.3
                self.noise.max_sigma = 0.3

            state = self.environment.reset()

            self.noise.reset()
            total_episode_reward = 0
            self.timestamp = 0

            # prvi red (i = 0) je gore sluzio za inicijalno stanje
            # trenutni red se koristi da se dobave scaling vrijednosti za sljedeci timestep
            # za i = len(df_train_day) ce next_state biti None, done = True, ali i tada hocemo da odradimo environment.step
            for next_time_step_idx in range(1, 25):
                state = np.asarray(state)
                action = self._get_action(state)
                action = self.noise.get_action(action, self.timestamp)
                if abs(action) > 1.0:
                    print('Warning: ddpg_lite.py.train - abs(action) > 1')

                next_state, reward, done, _, _ = self.environment.step(action)
                total_episode_reward += reward
                self.timestamp += 1

                self.replay_buffer.push(state, action, reward, next_state, done)
                if len(self.replay_buffer) > self.batch_size:
                    self._update()

                state = next_state

                if next_time_step_idx != self.environment.time_step:
                    print('Warning: ddpg_lite.py.train - something may be wrong with timestep indexing')

            if i_episode == 0:
                self.moving_average = total_episode_reward

            self.moving_average = 0.9 * self.moving_average + 0.1 * total_episode_reward
            total_episode_rewards.append(self.moving_average)

            if i_episode % 50 == 0:
                print("total_episode_reward: ", total_episode_reward)

            if i_episode % 50 == 0:
                torch.save(self.actor.state_dict(), "./trained_nets/model_actor" + str(i_episode))

        torch.save(self.actor.state_dict(), "model_actor")
        torch.save(self.critic.state_dict(), "model_critic")

        x_axis = [1 + j for j in range(len(total_episode_rewards))]
        plt.plot(x_axis, total_episode_rewards)
        plt.xlabel('Episode number')
        plt.ylabel('Total episode reward')
        plt.savefig("total_episode_rewards.png")
        plt.show()

    def test(self):
        print('agent testing started')
        self.actor.load_state_dict(torch.load("model_actor"))
        self.actor.eval()

        state = self.environment.reset()
        total_episode_reward = 0
        proposed_storage_powers = []
        actual_storage_powers = []
        storage_socs = []

        for next_time_step_idx in range(1, 25):
            state = np.asarray(state)
            action = self._get_action(state)
            proposed_storage_powers.append(action)
            if abs(action) > 1.0:
                print('Warning: deep_q_learning.train - abs(action) > 1')

            next_state, reward, done, actual_action, initial_soc = self.environment.step(action=action)

            actual_storage_powers.append(actual_action)
            storage_socs.append(initial_soc)

            total_episode_reward += reward
            state = next_state

        print('total_episode_reward', total_episode_reward)
        plot_daily_results(24, proposed_storage_powers, actual_storage_powers, storage_socs,
                           self.environment.model_data_provider.get_electricity_price())

    def _update(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        # output from actor network is normalized so:
        next_actions = reverse_action_tensor(next_actions, self.environment.action_space)

        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + (1.0 - dones) * self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime.detach())

        # actor loss
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
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def _get_action(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        self.actor.eval()
        action = self.actor.forward(state)
        self.actor.train()
        action = action.detach().cpu().numpy()[0, 0]
        # output from actor network is normalized so:
        action = reverse_action(action, self.environment.action_space)
        return action
