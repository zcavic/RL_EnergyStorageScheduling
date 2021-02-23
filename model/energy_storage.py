import math


class EnergyStorage:

    def __init__(self, max_p_mw, max_e_mwh, initial_power=0.0, initial_soc=0.0, capacity_fade=0.0):

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

    def capacity(self):
        return self.max_e_mwh - (self.max_e_mwh * self.capacity_fade)

    def get_power(self):
        return self.energyStorageState.power

    def send_action(self, action):

        if self._is_discharging(self._get_action(action)):
            self.energyStorageState.discharge(self._get_action(action))
        elif self._is_charging(self._get_action(action)):
            self.energyStorageState.charge(self._get_action(action))
        else:
            self.energyStorageState.turn_off()

        can_execute = True
        actual_action = self.energyStorageState.power / self.max_p_mw
        if self.energyStorageState.power == 0 and action != 0:
            can_execute = False

        self.energyStorageState.update_soc()

        return actual_action, can_execute

    # positive action is charging
    # with eps we check overcharging or overdischarging
    def _is_charging(self, action, eps=0.05):
        if action > 0 and (self.energyStorageState.soc + (action / self.capacity()) - eps) < 1:
            return True
        else:
            return False

    # negative action is discharging
    # with eps we check overcharging or overdischarging
    def _is_discharging(self, action, eps=0.05):
        if action < 0 and (self.energyStorageState.soc - (action / self.capacity()) + eps) > 0:
            return True
        else:
            return False

    def _get_action(self, action):
        if abs(action) > self.max_p_mw:
            return math.copysign(self.max_p_mw, action)
        else:
            return action


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
        if self.soc + (self.power / self._energy_storage.capacity()) > 1:
            self.soc = 1
            self._energy_storage.energyStorageState = IdleState(self._energy_storage, self.soc)
        elif self.soc + (self.power / self._energy_storage.capacity()) < 0:
            self.soc = 0
            self._energy_storage.energyStorageState = IdleState(self._energy_storage, self.soc)
        else:
            self.soc = self.soc + (self.power / self._energy_storage.capacity())


class IdleState(EnergyStorageState):

    # command for charge
    def charge(self, power):
        if self.soc < 1:
            self._energy_storage.energyStorageState = ChargingState(self._energy_storage, self.soc)
            self._energy_storage.energyStorageState.set_power(power)

    # command for discharge
    def discharge(self, power):
        if self.soc > 0:
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
            self._energy_storage.energyStorageState = DischargingState(self._energy_storage, self.soc)
            self._energy_storage.energyStorageState.set_power(power)
        else:
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


def _test_energy_storage1():
    es = EnergyStorage(max_p_mw=1, max_e_mwh=4, initial_soc=0)
    check = 0
    es.send_action(1)
    if math.isclose(es.energyStorageState.soc, 0.25) and math.isclose(es.energyStorageState.power, 1) and type(
            es.energyStorageState) == ChargingState:
        check += 1
    es.send_action(1)
    if math.isclose(es.energyStorageState.soc, 0.5) and math.isclose(es.energyStorageState.power, 1) and type(
            es.energyStorageState) == ChargingState:
        check += 1
    es.send_action(1)
    if math.isclose(es.energyStorageState.soc, 0.75) and math.isclose(es.energyStorageState.power, 1) and type(
            es.energyStorageState) == ChargingState:
        check += 1
    es.send_action(1)
    if math.isclose(es.energyStorageState.soc, 1) and math.isclose(es.energyStorageState.power, 1) and type(
            es.energyStorageState) == ChargingState:
        check += 1
    es.send_action(1)
    if math.isclose(es.energyStorageState.soc, 1) and math.isclose(es.energyStorageState.power, 0) and type(
            es.energyStorageState) == IdleState:
        check += 1
    es.send_action(-1)
    if math.isclose(es.energyStorageState.soc, 0.75) and math.isclose(es.energyStorageState.power, -1) and type(
            es.energyStorageState) == DischargingState:
        check += 1
    es.send_action(-1)
    if math.isclose(es.energyStorageState.soc, 0.5) and math.isclose(es.energyStorageState.power, -1) and type(
            es.energyStorageState) == DischargingState:
        check += 1
    es.send_action(-1)
    if math.isclose(es.energyStorageState.soc, 0.25) and math.isclose(es.energyStorageState.power, -1) and type(
            es.energyStorageState) == DischargingState:
        check += 1
    es.send_action(-1)
    if math.isclose(es.energyStorageState.soc, 0) and math.isclose(es.energyStorageState.power, -1) and type(
            es.energyStorageState) == DischargingState:
        check += 1
    if check == 9:
        print("TEST 1: OK")
    else:
        print("TEST 1: FAIL")


def _test_energy_storage_2():
    es = EnergyStorage(max_p_mw=1, max_e_mwh=5, initial_soc=0.83)
    es.send_action(1.2)
    if math.isclose(es.energyStorageState.soc, 1) and math.isclose(es.energyStorageState.power, 0) and type(
            es.energyStorageState) == IdleState:
        print("Test 2: OK")
    else:
        print("Test 2: FAIL")


def _test_energy_storage_3():
    es = EnergyStorage(max_p_mw=1, max_e_mwh=5, initial_soc=0.18)
    es.send_action(-1.2)
    if math.isclose(es.energyStorageState.soc, 0) and math.isclose(es.energyStorageState.power, 0) and type(
            es.energyStorageState) == IdleState:
        print("Test 3: OK")
    else:
        print("Test 3: FAIL")


def _test_energy_storage_4():
    es = EnergyStorage(max_p_mw=1, max_e_mwh=5, initial_soc=0.5)
    es.send_action(0.5)
    es.send_action(-0.5)
    if math.isclose(es.energyStorageState.soc, 0.5) and math.isclose(es.energyStorageState.power, -0.5) and type(
            es.energyStorageState) == DischargingState:
        print("Test 4: OK")
    else:
        print("Test 4: FAIL")
