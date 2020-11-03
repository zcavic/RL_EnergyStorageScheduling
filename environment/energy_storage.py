from enum import Enum
import math


class EnergyStorage:

    def __init__(self, index, power_grid, power_flow, initial_power=0, initial_soc=1, days_in_idle=0, no_of_cycles=0):
        self.id = index
        self.power_grid = power_grid
        self.power_flow = power_flow
        self.max_p_kw = self.power_grid.storage.p_mw.loc[index]
        self.max_e_mwh = self.power_grid.storage.max_e_mwh.loc[index]
        if initial_power < 0:
            self.energyStorageState = DischargingState(self, initial_soc, days_in_idle, no_of_cycles)
            self.energyStorageState.set_power(initial_power)
        elif initial_power > 0:
            self.energyStorageState = ChargingState(self, initial_soc, days_in_idle, no_of_cycles)
            self.energyStorageState.set_power(initial_power)
        else:
            self.energyStorageState = IdleState(self, initial_soc, days_in_idle, no_of_cycles)

    def state(self):
        return self.energyStorageState.state()

    def get_power(self):
        return self.energyStorageState.power

    def send_action(self, action):
        # chek for overcharging or overdischarging
        eps = 0.01
        if action - eps < -self.energyStorageState.soc or \
                action + eps > (self.max_e_mwh - self.energyStorageState.soc):
            self.energyStorageState.turn_off()
        # negative action is discharging
        elif action < 0:
            self.energyStorageState.discharge(action)
        # positive action is charging
        elif action > 0:
            self.energyStorageState.charge(action)
        # 0 is for turn off
        else:
            self.energyStorageState.turn_off()

        can_execute = True
        actual_action = self.energyStorageState.power
        if actual_action == 0 and action != 0:
            can_execute = False

        self.energyStorageState.update_soc()
        self.power_grid.storage.scaling.loc[self.id] = self.energyStorageState.power / self.max_p_kw
        self.power_flow.calculate_power_flow()

        return actual_action, can_execute


class EnergyStorageState:

    def __init__(self, energy_storage, initial_soc, days_in_idle, no_of_cycles):
        self._energy_storage = energy_storage
        self.soc = initial_soc
        self.power = 0
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

    def update_soc(self):
        return

    def set_power(self, power):
        if self._energy_storage.max_p_kw < abs(power):
            self.power = self._energy_storage.max_p_kw * math.copysign(1, power)
        else:
            self.power = power

    def get_actual_max_capacity(self):
        return self._energy_storage.max_e_mwh * (1 - self.get_cycle_fade() - self.get_idle_fade())

    def get_cycle_fade(self):
        soc = self.soc / self._energy_storage.max_e_mwh
        dod = (1 - soc)
        return 0.00568 * math.exp(-1.943 * soc) * (dod ** 0.7162) * math.sqrt(self.no_of_cycles)

    def get_idle_fade(self):
        soc = self.soc / self._energy_storage.max_e_mwh
        return 0.000112 * math.exp(0.7388 * soc) * (self.days_in_idle ** 0.8)


class IdleState(EnergyStorageState):

    def state(self):
        return State.IDLE

    # command for charge
    def charge(self, power):
        if self.soc < self.get_actual_max_capacity():
            # print('Idle state: start charging.')
            self._energy_storage.energyStorageState = ChargingState(self._energy_storage, self.soc,
                                                                    self.days_in_idle, self.no_of_cycles)
            self._energy_storage.energyStorageState.set_power(power)

    # command for discharge
    def discharge(self, power):
        if self.soc != 0:
            # print('Idle state: start discharging.')
            self._energy_storage.energyStorageState = DischargingState(self._energy_storage, self.soc,
                                                                       self.days_in_idle, self.no_of_cycles)
            self._energy_storage.energyStorageState.set_power(power)

    def turn_off(self):
        self.set_power(0)

    def update_soc(self):
        self.days_in_idle = self.days_in_idle + 1 / 24


class ChargingState(EnergyStorageState):

    def state(self):
        return State.CHARGING

    def charge(self, power):
        self._energy_storage.energyStorageState.set_power(power)

    # command for charge
    def discharge(self, power):
        if self.soc > 0:
            # print('Charging state: start discharging.')
            self._energy_storage.energyStorageState = DischargingState(self._energy_storage, self.soc,
                                                                       self.days_in_idle, self.no_of_cycles)
            self._energy_storage.energyStorageState.set_power(power)
        else:
            # print('Charging state: energy storage is empty and cant be more discharged.')
            self.turn_off()

    # turn off energy storage
    def turn_off(self):
        self._energy_storage.energyStorageState = IdleState(self._energy_storage,
                                                            self.soc, self.days_in_idle,
                                                            self.no_of_cycles)
        # print('Charging state: energy storage turned off.')

    # 1h elapsed
    def update_soc(self):
        if self.soc + self.power >= self.get_actual_max_capacity():
            self.soc = self.get_actual_max_capacity()
            self._energy_storage.energyStorageState = IdleState(self._energy_storage,
                                                                self.soc, self.days_in_idle,
                                                                self.no_of_cycles)
        else:
            self.soc = self.soc + self.power
        self.no_of_cycles += abs(self.power / self._energy_storage.max_e_mwh) / 2


class DischargingState(EnergyStorageState):

    def state(self):
        return State.DISCHARGING

    # command for discharge
    def charge(self, power):
        if self.soc < self.get_actual_max_capacity():
            self._energy_storage.energyStorageState = ChargingState(self._energy_storage,
                                                                    self.soc, self.days_in_idle,
                                                                    self.no_of_cycles)
            self._energy_storage.energyStorageState.set_power(power)
            # print('Discharging state: stat charging.')
        else:
            # print('Discharging state: energy storage is full and cant be more charged.')
            self.turn_off()

    def discharge(self, power):
        # print('Discharging state: discharging power updated.')
        self._energy_storage.energyStorageState.set_power(power)

    # turn off energy storage
    def turn_off(self):
        self._energy_storage.energyStorageState = IdleState(self._energy_storage,
                                                            self.soc, self.days_in_idle,
                                                            self.no_of_cycles)
        # print('Discharging state: energy storage turned off.')

    # 1h elapsed
    def update_soc(self):
        if self.soc + self.power <= 0:
            self.soc = 0
            self._energy_storage.energyStorageState = IdleState(self._energy_storage,
                                                                self.soc, self.days_in_idle,
                                                                self.no_of_cycles)
        else:
            self.soc = self.soc + self.power
        self.no_of_cycles += abs(self.power / self._energy_storage.max_e_mwh) / 2


class State(Enum):
    IDLE = 1
    CHARGING = 2
    DISCHARGING = 3
