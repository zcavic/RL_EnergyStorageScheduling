from enum import Enum


class EnergyStorage:

    def __init__(self, index, power_grid):
        self.index = index
        self.power_grid = power_grid
        self.max_p_kw = self.power_grid.storage.p_mw.loc[self.index]
        self.max_e_mwh = self.power_grid.storage.max_e_mwh.loc[self.index]
        self.state = IdleState(self, self.max_e_mwh, self.max_p_kw, 0)

    def state(self):
        return self.state.state()

    def get_power(self):
        return self.state.power

    def send_action(self, action):
        self.state.update_soc()
        self._update_state(action)
        self._update_scaling()

    def _charge(self, power):
        self.state.charge(power)

    def _discharge(self, power):
        self.state.discharge(power)

    def _turn_off(self):
        self.state.turn_off()

    def _update_state(self, action):
        if action < -self.state.soc or \
                action > (self.state.capacity - self.state.soc):
            self.state.turn_off()
        if action < 0:
            self.state.discharge(action)
        elif action > 0:
            self.state.charge(action)
        else:
            self.state.turn_off()

    def _update_scaling(self):
        self.power_grid.storage.scaling.loc[self.index] = self.state.power / self.max_p_kw


class State:

    def __init__(self, energy_storage, capacity, max_power, soc):
        self._energy_storage = energy_storage
        self.capacity = capacity
        self.max_power = max_power
        self.soc = soc
        self.power = 0

    def state(self):
        return

    def charge(self, power):
        return

    def discharge(self, power):
        return

    def turn_off(self):
        self._set_power(0)

    def update_soc(self):
        return

    def _set_power(self, power):
        if self.max_power < power:
            self.power = self.max_power
        else:
            self.power = power


class IdleState(State):

    def state(self):
        return EnergyStorageState.IDLE

    # command for charge
    def charge(self, power):
        print('Idle state: start charging.')
        self._energy_storage.state = ChargingState(self._energy_storage, self.capacity, self.max_power, self.soc)
        self._set_power(power)

    # command for discharge
    def discharge(self, power):
        print('Idle state: start discharging.')
        self._energy_storage.state = DischargingState(self._energy_storage, self.capacity, self.max_power, self.soc)
        self._set_power(power)


class ChargingState(State):

    def state(self):
        return EnergyStorageState.CHARGING

    # command for charge
    def discharge(self, power):
        print('Charging state: start discharging.')
        self._energy_storage.state = DischargingState(self._energy_storage, self.capacity, self.max_power, self.soc)
        self._energy_storage.state._set_power(power)

    def charge(self, power):
        self._set_power(power)

    # turn off energy storage
    def turn_off(self):
        self._energy_storage.state = IdleState(self._energy_storage, self.capacity, self.max_power, self.soc)
        self._set_power(0)
        print('Charging state: energy storage turned off.')

    # 1h elapsed
    def _update_soc(self):
        self.soc += self.power
        if self.capacity <= self.soc:
            self.soc = self.capacity
            self._energy_storage.state = IdleState(self._energy_storage, self.capacity, self.max_power, self.soc)
            print('Charging state: Energy storage is full.')
        else:
            print('Charging state: Energy storage is charging. Current capacity: ', self.soc)


class DischargingState(State):

    def state(self):
        return EnergyStorageState.DISCHARGING

    # command for discharge
    def charge(self, power):
        self._energy_storage.state = ChargingState(self._energy_storage, self.capacity, self.max_power, self.soc)
        self._energy_storage.state._set_power(power)
        print('Discharging state: stat charging.')

    def discharge(self, power):
        self._set_power(power)

    # turn off energy storage
    def turn_off(self):
        self._energy_storage.state = IdleState(self._energy_storage, self.capacity, self.max_power, self.soc)
        self._set_power(0)
        print('Discharging state: energy storage turned off.')

    # 1h elapsed
    def update_soc(self):
        self.soc -= self.power
        if self.soc <= 0:
            self.soc = 0
            self._energy_storage.state = IdleState(self._energy_storage, self.capacity, self.max_power, self.soc)
            print('Discharging state: Energy storage is empty.')
        else:
            print('Discharging state: Energy storage is discharging. Current capacity: ', self.soc)


class EnergyStorageState(Enum):
    IDLE = 1
    CHARGING = 2
    DISCHARGING = 3
