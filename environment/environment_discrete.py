import gym
from gym import spaces
import random
import numpy as np 
from power_algorithms.power_flow import PowerFlow
import power_algorithms.network_management as nm

#todo uvazi consumption u ovom fajlu gdje god se pominje
class EnvironmentDiscrete(gym.Env):
    
    def __init__(self):
        super(EnvironmentDiscrete, self).__init__()
        
        self.state = []

        self.network_manager = nm.NetworkManagement()
        self.power_flow = PowerFlow(self.network_manager)
        self.power_flow.calculate_power_flow() #potrebno zbog odredjivanja state_space_dims

        self.state_space_dims = len(self.power_flow.get_bus_voltages())
        self.n_actions = len(self.network_manager.get_all_capacitors())
        self.n_consumers = len(self.network_manager.power_grid.load.index)

    def _update_state(self):
        self.power_flow.calculate_power_flow()

        #bus_voltages_dict = self.power_flow.get_bus_voltages()
        #self.state = list(bus_voltages_dict.values())

        #line_rated_powers_dict = self.power_flow.get_line_rated_powers()
        #self.state = list(line_rated_powers_dict.values())

        return self.state

    #action: 0..n_actions
    def step(self, action):
        
        #next_state = self._update_state()

        #reward = self.calculate_reward(action)

        pass
        #return next_state, reward, done


    def calculate_reward(self, action):
        #racuna razliku u losses?
        pass
        #return reward

    def reset(self, consumption_percents):
        pass
        #return self.state