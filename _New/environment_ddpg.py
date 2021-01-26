import gym
import numpy as np
from abc import ABC
from gym import spaces
from _New.electricity_price_provider import get_electricity_price_for
from _New.energy_storage_factory import create_energy_storage, create_energy_storage_from_dataset


class EnvironmentDDPG(gym.Env, ABC):

    def __init__(self):
        super(EnvironmentDDPG, self).__init__()
        self.energy_storage = create_energy_storage()
        self._define_init_state()
        self._define_action_space()

    def step(self, action):
        initial_soc = self.energy_storage.energyStorageState.soc
        actual_action, can_execute = self.energy_storage.send_action(action)
        next_state = self._update_state()
        done = self.time_step == 24
        reward = self.calculate_reward(actual_action, can_execute)
        return next_state, reward, done, actual_action, initial_soc

    def calculate_reward(self, actual_action, can_execute):
        if can_execute:
            return get_electricity_price_for(self.time_step) * actual_action
        else:
            return -10

    def reset(self, dataset_row):
        self.energy_storage = create_energy_storage_from_dataset(dataset_row)
        self._define_init_state()
        return self.state

    def _update_state(self):
        self.state = []
        self.time_step += 1
        self.state.append(self.time_step / 25.0)
        self.state.append(self.energy_storage.energyStorageState.soc)

        if self.state_space_dims != len(self.state):
            print("Error: environment_ddpg.py -> _update_state - wrong state size")
        return self.state

    def _define_action_space(self):
        self.action_space_dims = 1  # todo broj energy storage-a umjesto hardkodovanja
        # storage actions:
        # p > 0 - charging
        # p < 0 - discharging
        self.low_set_point = -1.0  # p.u.
        self.high_set_point = 1.0
        # todo liste ispod ce se morati prosiriti kada bude bilo vise agenata (npr lista [self.low_set_point])
        self.action_space = spaces.Box(low=np.array([self.low_set_point]), high=np.array([self.high_set_point]),
                                       dtype=np.float16)
        if self.action_space_dims != self.action_space.shape[0]:
            print('Error: environment_ddpg.py -> _define_action_space - wrong action space size')

    def _define_init_state(self):
        self.state = []
        self.time_step = 0  # 0..23, ako je 24, onda ce next_state biti None
        self.agent_index = 0  # za sada imamo jednog agenta
        self.state.append(self.time_step / 25.0)
        self.state.append(self.energy_storage.energyStorageState.soc)
        self.state_space_dims = len(self.state)
