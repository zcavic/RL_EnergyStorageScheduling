import gym
from gym import spaces
import numpy as np
from obsolete.power_algorithms.power_flow import PowerFlow
import obsolete.power_algorithms.network_management as nm
from gym.spaces.space import Space
from obsolete.environment.energy_storage import EnergyStorage
# Custom space
from utils import get_scaling_from_row, get_energy_storage_state


class Incremental(Space):
    def __init__(self, start, stop, step, **kwargs):
        self.step = step
        self.size = int((stop - start) / step) + 1
        self.values = np.linspace(start, stop, self.size, **kwargs)
        super().__init__(self.values.shape, self.values.dtype)

    def sample(self):
        return np.random.choice(self.values)

    def contains(self, x):
        return x in self.values


class EnvironmentContinous(gym.Env):

    def __init__(self):
        super(EnvironmentContinous, self).__init__()

        self.state = []
        self.timestep = 0  # 0..23, ako je 24, onda ce next_state biti None

        # todo kada bude vise agenata ovo ce biti lista
        self.agent_index = 0  # za sada imamo jednog agenta

        self.network_manager = nm.NetworkManagement()
        self.power_flow = PowerFlow(self.network_manager)
        self.power_flow.calculate_power_flow()  # potrebno zbog odredjivanja state_space_dims
        self.losses_baseline = self.power_flow.get_losses()

        # p i q po granama je dovoljno da agent moze da napravi internu reprezentaciju gubitakam injektiranih snaga u cvorove
        # i snage koja se uzima iz prenosne mreze
        line_p_dict = self.power_flow.get_lines_active_power()
        line_q_dict = self.power_flow.get_lines_reactive_power()
        self.state.append(self.timestep / 25.0)
        # todo odabrati neku baznu snagu koja je priblizna najvecoj snazi prve sekcije u najgorem slucaju
        self.base_power = 0.01
        self.state += [val / self.base_power for val in list(line_p_dict.values())]  # moze ovo elegantnije
        self.state += [val / self.base_power for val in list(line_q_dict.values())]
        self.state.append(0.0)  # state of charge, prava vrijednost se postavlja ispod...

        self.state_space_dims = len(self.state)
        self.action_space_dims = 1  # todo iz pandapowera dobavi broj energy storage-a umjesto hardkodovanja

        # storage actions:
        # p > 0 - charging
        # p < 0 - discharging
        self.low_set_point = -1.0  # p.u.
        self.high_set_point = 1.0
        # todo liste ispod ce se morati prosiriti kada bude bilo vise agenata (npr lista [self.low_set_point])
        self.action_space = spaces.Box(low=np.array([self.low_set_point]), high=np.array([self.high_set_point]),
                                       dtype=np.float16)
        if (self.action_space_dims != self.action_space.shape[0]):
            print('Error in environment_continous.py: self.action_space_dims != self.action_space.shape[0]')

    def _update_state(self):
        self.state = []
        self.power_flow.calculate_power_flow()
        self.timestep += 1
        line_p_dict = self.power_flow.get_lines_active_power()
        line_q_dict = self.power_flow.get_lines_reactive_power()
        self.state.append(self.timestep / 25.0)
        self.state += [val / self.base_power for val in list(line_p_dict.values())]
        self.state += [val / self.base_power for val in list(line_q_dict.values())]
        self.state.append(self.energy_storage.energyStorageState.soc)

        if self.state_space_dims != len(self.state):
            print("Warning: environment_continous -> _update_state - wrong state size")
        return self.state

    def step(self, action, solar_percents, load_percents, electricity_price):
        initial_soc = self.energy_storage.energyStorageState.soc
        actual_action, can_execute, capacity_fade_delta = self.energy_storage.send_action(action[0])
        # self.network_manager.set_storage_scaling(action, self.agent_index)

        next_state = self._update_state()
        done = self.timestep == 24
        start = self.timestep == 1

        reward = self.calculate_reward(actual_action, can_execute, capacity_fade_delta)


        # sljedeci trenutak
        self.network_manager.set_scaling_to_all_generation(solar_percents[0])
        self.network_manager.set_scaling_to_all_load(load_percents[0])
        self.electricity_price_this_moment = electricity_price[0]
        return next_state, reward, done, actual_action, initial_soc

    def calculate_reward(self, actual_action, can_execute, capacity_fade_delta):
        if can_execute:
            electricity_price = self.electricity_price_this_moment * actual_action
            capacity_fade = capacity_fade_delta * 1000
            if electricity_price == 0:
                print('electricity price reward: ', electricity_price, 'capacity fade reward: ', capacity_fade)
            return - (electricity_price + capacity_fade)
        else:
            return - 10

    def reset(self, first_row):
        solar_percents, load_percents, electricity_price = get_scaling_from_row(first_row)
        self.timestep = 0
        self.electricity_price_this_moment = electricity_price[0]
        self.network_manager.set_storage_scaling(1.0, self.agent_index)

        self.network_manager.set_scaling_to_all_generation(solar_percents[0])
        self.network_manager.set_scaling_to_all_load(load_percents[0])

        # todo neka ovo bude lista ili nesto...?
        state_of_charge, days_in_idle, no_of_cycles, storage_load = get_energy_storage_state(first_row)
        index = self.network_manager.get_es_indexes()  # za sada upravljamo samo jednim ES
        self.energy_storage = EnergyStorage(index[0], self.network_manager.power_grid, self.power_flow, storage_load,
                                            state_of_charge, days_in_idle, no_of_cycles)

        self.state = []
        self.power_flow.calculate_power_flow()
        line_p_dict = self.power_flow.get_lines_active_power()
        line_q_dict = self.power_flow.get_lines_reactive_power()
        self.state.append(self.timestep / 25.0)
        self.state += [val / self.base_power for val in list(line_p_dict.values())]
        self.state += [val / self.base_power for val in list(line_q_dict.values())]
        self.state.append(self.energy_storage.energyStorageState.soc)

        return self.state
