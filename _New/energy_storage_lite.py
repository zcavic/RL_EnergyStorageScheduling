import math


class EnergyStorageLite:

    def __init__(self, max_p_mw, max_e_mwh, initial_power=0, initial_soc=0, capacity_fade=0):

        self.max_p_mw = max_p_mw
        self.max_e_mwh = max_e_mwh
        self.capacity_fade = capacity_fade
        if initial_power < 0:
            self.energyStorageState = DischargingState(self, initial_soc)
            self.energyStorageState.set_power(initial_power)
        elif initial_power > 0:
            self.energyStorageState = ChargingState(self, initial_soc)
            self.energyStorageState.set_power(initial_power)
        else:
            self.energyStorageState = IdleState(self, initial_soc)

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
        actual_action = self.energyStorageState.power / self.max_p_mw
        if actual_action == 0 and action != 0:
            can_execute = False

        self.energyStorageState.update_soc()

        return actual_action, can_execute


class EnergyStorageState:

    def __init__(self, energy_storage, initial_soc):
        self._energy_storage = energy_storage
        self.soc = initial_soc
        self.power = 0

    def charge(self, power):
        return

    def discharge(self, power):
        return

    def turn_off(self):
        return

    def update_soc(self):
        return

    def update_capacity_fade(self):
        return

    def set_power(self, power):
        if self._energy_storage.max_p_mw < abs(power):
            self.power = self._energy_storage.max_p_mw * math.copysign(1, power)
        else:
            self.power = power

    # 1h elapsed
    def update_soc(self):
        if self.soc + (self.power / self._energy_storage.max_e_mwh) > 1:
            self.soc = 1
            self._energy_storage.energyStorageState = IdleState(self._energy_storage, self.soc)
        elif self.soc + (self.power / self._energy_storage.max_e_mwh) < 0:
            self.soc = 0
            self._energy_storage.energyStorageState = IdleState(self._energy_storage, self.soc)
        else:
            self.soc = self.soc + (self.power / self._energy_storage.max_e_mwh)


class IdleState(EnergyStorageState):

    # command for charge
    def charge(self, power):
        if self.soc + (self.power / self._energy_storage.max_e_mwh) < 1:
            self._energy_storage.energyStorageState = ChargingState(self._energy_storage, self.soc)
            self._energy_storage.energyStorageState.set_power(power)

    # command for discharge
    def discharge(self, power):
        if self.soc != 0:
            # print('Idle state: start discharging.')
            self._energy_storage.energyStorageState = DischargingState(self._energy_storage, self.soc)
            self._energy_storage.energyStorageState.set_power(power)

    def turn_off(self):
        self.set_power(0)


class ChargingState(EnergyStorageState):

    def charge(self, power):
        self._energy_storage.energyStorageState.set_power(power)

    # command for charge
    def discharge(self, power):
        if self.soc > 0:
            # print('Charging state: start discharging.')
            self._energy_storage.energyStorageState = DischargingState(self._energy_storage, self.soc)
            self._energy_storage.energyStorageState.set_power(power)
        else:
            # print('Charging state: energy storage is empty and cant be more discharged.')
            self.turn_off()

    # turn off energy storage
    def turn_off(self):
        self._energy_storage.energyStorageState = IdleState(self._energy_storage, self.soc)


class DischargingState(EnergyStorageState):

    # command for discharge
    def charge(self, power):
        if self.soc < 1:
            self._energy_storage.energyStorageState = ChargingState(self._energy_storage, self.soc)
            self._energy_storage.energyStorageState.set_power(power)
        else:
            self.turn_off()

    def discharge(self, power):
        self._energy_storage.energyStorageState.set_power(power)

    # turn off energy storage
    def turn_off(self):
        self._energy_storage.energyStorageState = IdleState(self._energy_storage, self.soc)
