from enum import Enum


class EnergyStorage:

    def __init__(self, index, power_grid):
        self.id = index
        self.power_grid = power_grid
        self.max_p_kw = self.power_grid.storage.p_mw.loc[index]
        self.max_e_mwh = self.power_grid.storage.max_e_mwh.loc[index]
        self.energyStorageState = IdleState(self, self.max_e_mwh, self.max_p_kw, 0)

    def state(self):
        return self.energyStorageState.state()

    def charge(self, power):
        self.energyStorageState.charge(power)

    def discharge(self, power):
        self.energyStorageState.discharge(power)

    def turn_off(self):
        self.energyStorageState.turn_off()

    def get_power(self):
        return self.energyStorageState.power

    def send_action(self, action):
        self.energyStorageState.update_soc()
        if action < -self.energyStorageState.soc or \
                action > (self.energyStorageState.capacity - self.energyStorageState.soc):
            self.energy_storage.turn_off()
        if action < 0:
            self.energy_storage.discharge(action)
        elif action > 0:
            self.energy_storage.charge(action)
        else:
            self.energy_storage.turn_off()
        self.power_grid.storage.scaling.loc[self.index] = self.energyStorageState.power / self.max_p_kw


class EnergyStorageState:

    def __init__(self, energy_storage, capacity, max_power, current_capacity):
        self._energy_storage = energy_storage
        self.capacity = capacity
        self.max_power = max_power
        self.soc = current_capacity
        self.power = 0

    def state(self):
        return

    def charge(self, power):
        return

    def discharge(self, power):
        return

    def turn_off(self):
        return

    def _update_soc(self):
        return

    def _set_power(self, power):
        if self.max_power < power:
            self.power = self.max_power
        else:
            self.power = power


class IdleState(EnergyStorageState):

    def state(self):
        return State.IDLE

    # command for charge
    def charge(self, power):
        if self.capacity != self.soc:
            print('Idle state: start charging.')
            self._energy_storage.energyStorageState = ChargingState(self._energy_storage, self.capacity,
                                                                    self.max_power, self.soc)
            self._energy_storage.energyStorageState._set_power(power)

    # command for discharge
    def discharge(self, power):
        if self.soc != 0:
            print('Idle state: start discharging.')
            self._energy_storage.energyStorageState = DischargingState(self._energy_storage, self.capacity,
                                                                       self.max_power, self.soc)
            self._energy_storage.energyStorageState._set_power(power)


class ChargingState(EnergyStorageState):

    def state(self):
        return State.CHARGING

    # command for charge
    def discharge(self, power):
        if self.current_capacity != 0:
            print('Charging state: start discharging.')
            self._energy_storage.energyStorageState = DischargingState(self._energy_storage, self.capacity,
                                                                       self.max_power, self.current_capacity)
            self._energy_storage.energyStorageState._set_power(power)

    # turn off energy storage
    def turn_off(self):
        self._energy_storage.energyStorageState = IdleState(self._energy_storage, self.capacity, self.max_power,
                                                            self.current_capacity)
        print('Discharging state: energy storage turned off.')

    # 1h elapsed
    def _update_soc(self):
        self.current_capacity += self.power
        if self.capacity <= self.current_capacity:
            self.current_capacity = self.capacity
            self._energy_storage.energyStorageState = IdleState(self._energy_storage, self.capacity, self.max_power,
                                                                self.current_capacity)
            print('Charging state: Energy storage is full.')
        else:
            print('Charging state: Energy storage is charging. Current capacity: ', self.current_capacity)


class DischargingState(EnergyStorageState):

    def state(self):
        return State.DISCHARGING

    # command for discharge
    def charge(self, power):
        if self.soc < self.capacity:
            self._energy_storage.energyStorageState = ChargingState(self._energy_storage, self.capacity, self.max_power,
                                                                    self.soc)
            self._energy_storage.energyStorageState._set_power(power)
            print('Discharging state: stat charging.')

    # turn off energy storage
    def turn_off(self):
        self._energy_storage.energyStorageState = IdleState(self._energy_storage, self.capacity, self.max_power,
                                                            self.soc)
        print('Discharging state: energy storage turned off.')

    # 1h elapsed
    def _update_soc(self):
        self.soc -= self.power
        if self.soc <= 0:
            self.soc = 0
            self._energy_storage.energyStorageState = IdleState(self._energy_storage, self.capacity, self.max_power,
                                                                self.soc)
            print('Discharging state: Energy storage is empty.')
        else:
            print('Discharging state: Energy storage is discharging. Current capacity: ', self.soc)


class State(Enum):
    IDLE = 1
    CHARGING = 2
    DISCHARGING = 3
