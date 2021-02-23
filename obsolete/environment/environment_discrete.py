import gym
import numpy as np
from obsolete.power_algorithms.power_flow import PowerFlow
import obsolete.power_algorithms.network_management as nm
from gym.spaces.space import Space
from obsolete.environment.energy_storage import EnergyStorage

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
        self.startng_loss = self.power_flow.get_losses() #zbog calculate reward

        #p i q po granama je dovoljno da agent moze da napravi internu reprezentaciju gubitakam injektiranih snaga u cvorove
        # i snage koja se uzima iz prenosne mreze
        line_p_dict = self.power_flow.get_lines_active_power()
        line_q_dict = self.power_flow.get_lines_reactive_power()
        self.state.append(self.timestep / 25.0)
        self.base_power = 6.0
        self.state += [val / self.base_power for val in list(line_p_dict.values())]
        self.state += [val / self.base_power for val in list(line_q_dict.values())]

        self.state.append(0.0) #state of charge, prava vrijednost se postavlja ispod...

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
        self.state.append(self.timestep / 25.0)
        self.state += [val / self.base_power for val in list(line_p_dict.values())]
        self.state += [val / self.base_power for val in list(line_q_dict.values())]
        self.state.append(self.energy_storage.energyStorageState.soc)

        if self.state_space_dims != len(self.state):
            print("Warning: environment_distrete -> _update_state - wrong state size")
        return self.state

    def _update_available_actions(self):
        self.action_indices = [i for i in range(self.n_actions)]
        self.available_action_values = self.action_space.values.tolist()
        self.available_actions = dict(zip(self.action_indices, self.available_action_values))
        eps = 0.01

        for action_id in self.action_indices:
            action = self.available_actions[action_id]
            if action - eps <= -self.energy_storage.energyStorageState.soc or \
                    action + eps >= (self.energy_storage.energyStorageState.capacity - self.energy_storage.energyStorageState.soc):
                self.available_actions.pop(action_id)

    def step(self, action, solar_percents, load_percents):
        initial_soc = self.energy_storage.energyStorageState.soc
        actual_action, can_execute = self.energy_storage.send_action(action)
        #self.network_manager.set_storage_scaling(action, self.agent_index)

        self.network_manager.set_generation_scaling(solar_percents)
        self.network_manager.set_load_scaling(load_percents)

        next_state = self._update_state()
        reward = self.calculate_reward(action, actual_action, can_execute)
        done = self.timestep == 24
        self._update_available_actions()
        return next_state, reward, done, actual_action, initial_soc


    def calculate_reward(self, action, cant_execute):
        #za sada q ne razmatramo jer storage salje akcije samo po p
        #q idalje stoji u stanju, ako htjednemo da ukljucimo losses ovdje
        #ako je network_injected_p negativno, onda ce agent pokusavati da ga poveca - da poveca prodaju u prenosnu mrezu, to je ok
        #return -1 * self.power_flow.get_network_injected_p().values[0]
        
        #reward =  -0.1 * (self.power_flow.get_losses() - self.startng_loss)
        if self.timestep >= 7 and self.timestep < 15:
            reward = - 0.2 * actual_action
        else:
            reward = 0
            
        return reward

    def reset(self, solar_percents, load_percents):
        self.timestep = 0
        self.network_manager.set_storage_scaling(1.0, self.agent_index)

        self.network_manager.set_generation_scaling(solar_percents)
        self.network_manager.set_load_scaling(load_percents)
        
        #todo neka ovo bude lista ili nesto...?
        index = 0 #ili 1, provjeri?
        self.energy_storage = EnergyStorage(index, self.network_manager.power_grid, self.power_flow)

        self.state = []
        self.power_flow.calculate_power_flow()
        line_p_dict = self.power_flow.get_lines_active_power()
        line_q_dict = self.power_flow.get_lines_reactive_power()
        self.state.append(self.timestep / 25.0)
        self.state += [val / self.base_power for val in list(line_p_dict.values())]
        self.state += [val / self.base_power for val in list(line_q_dict.values())]
        self.state.append(self.energy_storage.energyStorageState.soc) #todo ovo kasnije mora biti lista a ne jedan soc

        self._update_available_actions()

        return self.state