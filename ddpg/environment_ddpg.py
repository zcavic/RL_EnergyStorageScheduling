import gym
import numpy as np
from abc import ABC
from gym import spaces
from model.model_data_provider import ModelDataProvider
from model.energy_storage_factory import create_energy_storage
from utils import select_random_day_start
from datetime import timedelta


class EnvironmentDDPG(gym.Env, ABC):

    def __init__(self, dataset):
        super(EnvironmentDDPG, self).__init__()
        self.current_datetime = None
        self.energy_storage = None
        self._define_init_state()
        self._define_action_space()
        self.model_data_provider = ModelDataProvider(dataset)

    def step(self, action):
        initial_soc = self.energy_storage.energyStorageState.soc
        power = action[0] * self.energy_storage.max_p_mw
        actual_action, can_execute = self.energy_storage.send_action(power)
        reward = self._calculate_reward(actual_action, can_execute)
        next_state = self._update_state()
        done = self.time_step == 24
        return next_state, reward, done, actual_action, initial_soc

    def reset(self, df):
        self.current_datetime = select_random_day_start(df)
        self.energy_storage = self.model_data_provider.create_energy_storage(self.current_datetime)
        self._define_init_state()
        return self.state

    def _calculate_reward(self, actual_action, can_execute):
        reward_scaling = 10
        reward_for_not_executed = -2
        if can_execute:
            return (-self.model_data_provider.get_electricity_price_for(
                self.current_datetime) * actual_action) / reward_scaling
        else:
            return reward_for_not_executed / reward_scaling

    def _update_state(self):
        self.state = []
        self.time_step += 1
        self.current_datetime += timedelta(hours=1)
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
        self.state.append(0)  # time step
        self.state.append(0)  # initial SoC
        self.state_space_dims = len(self.state)
