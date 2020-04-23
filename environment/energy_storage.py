from enum import Enum


class EnergyStorage:

    def __init__(self, index, power_grid, power_flow):
        self.id = index
        self.power_grid = power_grid
        self.power_flow = power_flow
        self.max_p_kw = self.power_grid.storage.p_mw.loc[index]
        #self.max_e_mwh = self.power_grid.storage.max_e_mwh.loc[index] #1000000.0
        self.max_e_mwh = 100000.0
        initial_soc = 0.0
        self.energyStorageState = IdleState(self, self.max_e_mwh, self.max_p_kw, initial_soc)

    def state(self):
        return self.energyStorageState.state()

    def get_power(self):
        return self.energyStorageState.power

    def send_action(self, action):
        actual_action = action[0]
        #print('Storage is in', self.state(), 'and action', action, 'will be executed.')
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

        self.energyStorageState.update_soc()
        self.power_grid.storage.scaling.loc[self.id] = self.energyStorageState.power / self.max_p_kw
        self.power_flow.calculate_power_flow()

        return actual_action, cant_execute


class EnergyStorageState:

    def __init__(self, energy_storage, capacity, max_power, initial_soc):
        self._energy_storage = energy_storage
        self.capacity = capacity
        self.max_power = max_power
        self.soc = initial_soc
        self.power = 0

    def state(self):
        return

    def charge(self, power):
        return

    def discharge(self, power):
        return

    def turn_off(self):
        return

    def update_soc(self):
        return

    def set_power(self, power):
        if self.max_power < power:
            self.power = [self.max_power] * len(power)
        else:
            self.power = power


class IdleState(EnergyStorageState):

    def state(self):
        return State.IDLE

    # command for charge
    def charge(self, power):
        if self.soc < self.capacity:
            #print('Idle state: start charging.')
            self._energy_storage.energyStorageState = ChargingState(self._energy_storage, self.capacity,
                                                                    self.max_power, self.soc)
            self._energy_storage.energyStorageState.set_power(power)

    # command for discharge
    def discharge(self, power):
        if self.soc != 0:
            #print('Idle state: start discharging.')
            self._energy_storage.energyStorageState = DischargingState(self._energy_storage, self.capacity,
                                                                       self.max_power, self.soc)
            self._energy_storage.energyStorageState.set_power(power)

    def turn_off(self):
        self.set_power(0)


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
                                                                       self.max_power, self.soc)
            self._energy_storage.energyStorageState.set_power(power)
        else:
            #print('Charging state: energy storage is empty and cant be more discharged.')
            self.turn_off()

    # turn off energy storage
    def turn_off(self):
        self._energy_storage.energyStorageState = IdleState(self._energy_storage, self.capacity, self.max_power,
                                                            self.soc)
        #print('Charging state: energy storage turned off.')

    # 1h elapsed
    def update_soc(self):
        self.soc += self.power[0]
        if self.soc >= self.capacity:
            self.soc = self.capacity
            self._energy_storage.energyStorageState = IdleState(self._energy_storage, self.capacity, self.max_power,
                                                                self.soc)
            #print('Charging state: Energy storage is full.')
        #else:
            #print('Charging state: Energy storage is charging. State of charge: ', self.soc)


class DischargingState(EnergyStorageState):

    def state(self):
        return State.DISCHARGING

    # command for discharge
    def charge(self, power):
        if self.soc < self.capacity:
            self._energy_storage.energyStorageState = ChargingState(self._energy_storage, self.capacity, self.max_power,
                                                                    self.soc)
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
                                                            self.soc)
        #print('Discharging state: energy storage turned off.')

    # 1h elapsed
    def update_soc(self):
        self.soc += self.power[0]
        if self.soc <= 0:
            self.soc = 0
            self._energy_storage.energyStorageState = IdleState(self._energy_storage, self.capacity, self.max_power,
                                                                self.soc)
            #print('Discharging state: Energy storage is empty.')
        #else:
            #print('Discharging state: Energy storage is discharging. State of charge: ', self.soc)


class State(Enum):
    IDLE = 1
    CHARGING = 2
    DISCHARGING = 3
