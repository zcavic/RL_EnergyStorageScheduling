import gym
from gym import spaces
import random
import numpy as np 
from power_algorithms.power_flow import PowerFlow
import power_algorithms.network_management as nm
from gym.spaces import Tuple
from gym.spaces.space import Space

#Custom space
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

class EnvironmentDiscrete(gym.Env):
    
    def __init__(self):
        super(EnvironmentDiscrete, self).__init__()
        
        self.state = []
        self.timestep = 0 #0..23, ako je 24, onda ce next_state biti None

        self.agent_index = 0 #za sada imamo jednog agenta

        self.network_manager = nm.NetworkManagement()
        self.power_flow = PowerFlow(self.network_manager)
        self.power_flow.calculate_power_flow() #potrebno zbog odredjivanja state_space_dims

        #p i q po granama je dovoljno da agent moze da napravi internu reprezentaciju gubitakam injektiranih snaga u cvorove
        # i snage koja se uzima iz prenosne mreze
        line_p_dict = self.power_flow.get_lines_active_power()
        line_q_dict = self.power_flow.get_lines_reactive_power()
        self.state.append(self.timestep * 1.0)
        self.state += list(line_p_dict.values())
        self.state += list(line_q_dict.values())

        self.state_space_dims = len(self.state)
        
        #storage actions:
        # p > 0 - charging
        # p < 0 - discharging
        self.low_set_point = -1.0 # p.u.
        self.high_set_point = 1.0
        self.action_space = Incremental(self.low_set_point, self.high_set_point, 0.1)
        self.n_actions = self.action_space.size

    def _update_state(self):
        self.state = []
        self.power_flow.calculate_power_flow()
        self.timestep += 1
        line_p_dict = self.power_flow.get_lines_active_power()
        line_q_dict = self.power_flow.get_lines_reactive_power()
        self.state.append(self.timestep * 1.0)
        self.state += list(line_p_dict.values())
        self.state += list(line_q_dict.values())

        if self.state_space_dims != len(self.state):
            print("Warning: environment_distrete -> _update_state - wrong state size")
        return self.state

    def step(self, action, solar_percents, load_percents):
        self.network_manager.set_storage_scaling(action, self.agent_index)

        self.network_manager.set_generation_scaling(solar_percents)
        self.network_manager.set_load_scaling(load_percents)

        next_state = self._update_state()
        reward = self.calculate_reward(action)
        done = self.timestep == 24
        return next_state, reward, done


    def calculate_reward(self, action):
        #za sada q ne razmatramo jer storage salje akcije samo po p
        #q idalje stoji u stanju, ako htjednemo da ukljucimo losses ovdje
        #ako je network_injected_p negativno, onda ce agent pokusavati da ga poveca - da poveca prodaju u prenosnu mrezu, to je ok
        return -1 * self.power_flow.get_network_injected_p().values[0]

    def reset(self, solar_percents, load_percents):
        self.timestep = 0
        self.network_manager.set_storage_scaling(1.0, self.agent_index)

        self.network_manager.set_generation_scaling(solar_percents)
        self.network_manager.set_load_scaling(load_percents)

        self.state = []
        self.power_flow.calculate_power_flow()
        line_p_dict = self.power_flow.get_lines_active_power()
        line_q_dict = self.power_flow.get_lines_reactive_power()
        self.state.append(self.timestep * 1.0)
        self.state += list(line_p_dict.values())
        self.state += list(line_q_dict.values())

        return self.state