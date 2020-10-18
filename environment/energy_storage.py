from enum import Enum
import math


class EnergyStorage:

    def __init__(self, index, power_grid, power_flow):
        self.id = index
        self.power_grid = power_grid
        self.power_flow = power_flow
        self.max_p_kw = self.power_grid.storage.p_mw.loc[index]
        self.max_e_mwh = self.power_grid.storage.max_e_mwh.loc[index] #1000000.0
        initial_soc = 4.0
        self.energyStorageState = IdleState(self, self.max_e_mwh, self.max_p_kw, initial_soc, 0, 0, 0)

    def state(self):
        return self.energyStorageState.state()

    def get_power(self):
        return self.energyStorageState.power

    def send_action(self, action, timestep):
        actual_action = action[0]
        cant_execute = False
        eps = 0.01
        if action - eps < -self.energyStorageState.soc or \
                action + eps > (self.energyStorageState.capacity - self.energyStorageState.soc):
            self.energyStorageState.turn_off()
            actual_action = 0
            cant_execute = True
        if action < 0:
            self.energyStorageState.discharge(action)
        elif action > 0:
            self.energyStorageState.charge(action)
        else:
            self.energyStorageState.turn_off()

        self.energyStorageState.update_soc(timestep)
        self.power_grid.storage.scaling.loc[self.id] = self.energyStorageState.power / self.max_p_kw
        self.power_flow.calculate_power_flow()

        return actual_action, cant_execute


class EnergyStorageState:

    def __init__(self, energy_storage, capacity, max_power, initial_soc, initial_timestep, days_in_idle, no_of_cycles):
        self._energy_storage = energy_storage
        self.capacity = capacity
        self.max_power = max_power
        self.soc = initial_soc
        self.power = 0
        self.timestep = initial_timestep
        self.days_in_idle = days_in_idle
        self.no_of_cycles = no_of_cycles

    def state(self):
        return

    def charge(self, power):
        return

    def discharge(self, power):
        return

    def turn_off(self):
        return

    def update_soc(self, timestep):
        return

    def set_power(self, power):
        if self.max_power < power:
            self.power = [self.max_power] * len(power)
        else:
            self.power = power

    def get_actual_max_capacity(self):
        soc = self.soc / self.capacity
        dod = (1 - soc)
        cycle = 0.00568 * math.exp(-1.943 * soc) * (dod ** 0.7162) * math.sqrt(self.no_of_cycles)
        idle = 0.000112 * math.exp(0.7388 * soc) * (self.days_in_idle ** 0.8)
        return self.capacity * (idle + cycle)


class IdleState(EnergyStorageState):

    def state(self):
        return State.IDLE

    # command for charge
    def charge(self, power):
        if self.soc < self.get_actual_max_capacity():
            #print('Idle state: start charging.')
            self._energy_storage.energyStorageState = ChargingState(self._energy_storage, self.capacity,
                                                                    self.max_power, self.soc, self.timestep,
                                                                    self.days_in_idle, self.no_of_cycles)
            self._energy_storage.energyStorageState.set_power(power)

    # command for discharge
    def discharge(self, power):
        if self.soc != 0:
            #print('Idle state: start discharging.')
            self._energy_storage.energyStorageState = DischargingState(self._energy_storage, self.capacity,
                                                                       self.max_power, self.soc, self.timestep,
                                                                       self.days_in_idle, self.no_of_cycles)
            self._energy_storage.energyStorageState.set_power(power)

    def turn_off(self):
        self.set_power(0)

    def update_soc(self, timestep):
        self.days_in_idle = self.days_in_idle + abs(timestep - self.timestep) / 24
        self.timestep = timestep


class ChargingState(EnergyStorageState):

    def state(self):
        return State.CHARGING

    def charge(self, power):
        self._energy_storage.energyStorageState.set_power(power)

    # command for charge
    def discharge(self, power):
        if self.soc > 0:
            #print('Charging state: start discharging.')
            self._energy_storage.energyStorageState = DischargingState(self._energy_storage, self.capacity,
                                                                       self.max_power, self.soc, self.timestep,
                                                                       self.days_in_idle, self.no_of_cycles)
            self._energy_storage.energyStorageState.set_power(power)
        else:
            #print('Charging state: energy storage is empty and cant be more discharged.')
            self.turn_off()

    # turn off energy storage
    def turn_off(self):
        self._energy_storage.energyStorageState = IdleState(self._energy_storage, self.capacity, self.max_power,
                                                            self.soc, self.timestep,self.days_in_idle,
                                                            self.no_of_cycles)
        #print('Charging state: energy storage turned off.')

    # 1h elapsed
    def update_soc(self, timestep):
        self.timestep = timestep
        self.no_of_cycles = self.no_of_cycles + (abs(self.power[0]) / self.capacity)
        self.soc += self.power[0]
        if self.soc >= self.get_actual_max_capacity():
            self.soc = self.get_actual_max_capacity()
            self._energy_storage.energyStorageState = IdleState(self._energy_storage, self.capacity, self.max_power,
                                                                self.soc, self.timestep, self.days_in_idle,
                                                                self.no_of_cycles)


class DischargingState(EnergyStorageState):

    def state(self):
        return State.DISCHARGING

    # command for discharge
    def charge(self, power):
        if self.soc < self.get_actual_max_capacity():
            self._energy_storage.energyStorageState = ChargingState(self._energy_storage, self.capacity, self.max_power,
                                                                    self.soc, self.timestep, self.days_in_idle,
                                                                    self.no_of_cycles)
            self._energy_storage.energyStorageState.set_power(power)
            #print('Discharging state: stat charging.')
        else:
            #print('Discharging state: energy storage is full and cant be more charged.')
            self.turn_off()

    def discharge(self, power):
        #print('Discharging state: discharging power updated.')
        self._energy_storage.energyStorageState.set_power(power)

    # turn off energy storage
    def turn_off(self):
        self._energy_storage.energyStorageState = IdleState(self._energy_storage, self.capacity, self.max_power,
                                                            self.soc, self.timestep, self.days_in_idle,
                                                            self.no_of_cycles)
        #print('Discharging state: energy storage turned off.')

    # 1h elapsed
    def update_soc(self, timestep):
        self.timestep = timestep
        self.no_of_cycles = self.no_of_cycles + abs(self.power[0] / self.capacity)
        self.soc += self.power[0]
        if self.soc <= 0:
            self.soc = 0
            self._energy_storage.energyStorageState = IdleState(self._energy_storage, self.capacity, self.max_power,
                                                                self.soc, self.timestep, self.days_in_idle,
                                                                self.no_of_cycles)


class State(Enum):
    IDLE = 1
    CHARGING = 2
    DISCHARGING = 3
