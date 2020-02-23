from environment.energy_storage import EnergyStorage
from power_algorithms.forecast import Forecast


def _get_storage_capacity_timeline(storage_actions_timeline: list):

    storage_capacity_timeline = [0] * len(storage_actions_timeline)
    for timestamp in range(len(storage_actions_timeline)):
        next_timestamp = timestamp
        while next_timestamp < len(storage_actions_timeline):
            storage_capacity_timeline[next_timestamp] = storage_capacity_timeline[next_timestamp] + \
                                                              storage_actions_timeline[timestamp]
            next_timestamp = next_timestamp + 1

    minimum = min(storage_capacity_timeline)
    if minimum < 0:
        for timestamp in range(len(storage_capacity_timeline)):
            storage_capacity_timeline[timestamp] = storage_capacity_timeline[timestamp] + \
                                                         abs(minimum)

    return storage_capacity_timeline


class HeuristicStorageScheduler(object):

    def __init__(self, energy_storage: EnergyStorage, forecast: Forecast):
        self._storage_power = energy_storage.energyStorageState.max_power
        self._storage_capacity = energy_storage.energyStorageState.capacity
        self._consumption = forecast.consumption.copy()
        self._old_consumption = forecast.consumption.copy()
        self._storage_actions_timeline = [0] * len(self._consumption)
        self._storage_capacity_timeline = [0] * len(self._consumption)
        self._step = 0.5  # this is step (percent of nominal power) for gradient charge or discharge of energy storage

        print('Old consumption: ', forecast.consumption)

    def calculate_storage_schedule(self):

        charge = True
        discharge = True
        iterator = 0
        while charge and discharge and iterator < 1000:
            draft_storage_actions_timeline = self._storage_actions_timeline.copy()
            charge = self._charge(_get_storage_capacity_timeline(draft_storage_actions_timeline),
                                  draft_storage_actions_timeline, self._consumption.copy())
            discharge = self._discharge(_get_storage_capacity_timeline(draft_storage_actions_timeline),
                                        draft_storage_actions_timeline, self._consumption.copy())

            if not charge or not discharge:
                break
            if not self._validate_action(draft_storage_actions_timeline):
                break

            self._storage_actions_timeline = draft_storage_actions_timeline
            self._update_storage_capacity_timeline()
            self._update_consumption_timeline()
            iterator = iterator + 1

        print('New Consumption: ', self._consumption)
        print('Actions:  ', self._storage_actions_timeline)
        print('Capacity: ', self._storage_capacity_timeline)

    def _charge(self, storage_capacity_timeline: list, storage_actions_timeline: list, consumption: list):

        # case when charge was tried at all hours
        if not consumption:
            return False

        timestamp = self._consumption.index(min(consumption))

        new_capacity = storage_capacity_timeline[timestamp] + (self._storage_power * self._step)
        new_action = storage_actions_timeline[timestamp] + (self._storage_power * self._step)
        if new_capacity <= self._storage_capacity and new_action <= self._storage_power:
            storage_actions_timeline[timestamp] = new_action
            return True
        else:
            del consumption[consumption.index(min(consumption))]
            return self._charge(storage_capacity_timeline, storage_actions_timeline, consumption)

    def _discharge(self, storage_capacity_timeline: list, storage_actions_timeline: list, consumption: list):

        # case when discharge was tried at all hours
        if not consumption:
            return False

        timestamp = self._consumption.index(max(consumption))

        new_capacity = storage_capacity_timeline[timestamp] - (self._storage_power * self._step)
        new_action = storage_actions_timeline[timestamp] - (self._storage_power * self._step)
        if new_capacity >= 0 and abs(new_action) <= self._storage_power:
            storage_actions_timeline[timestamp] = new_action
            return True
        else:
            del consumption[consumption.index(max(consumption))]
            return self._discharge(storage_capacity_timeline, storage_actions_timeline, consumption)

    def _validate_action(self, storage_actions_timeline: list):
        if max(_get_storage_capacity_timeline(storage_actions_timeline)) <= self._storage_capacity:
            return True
        else:
            return False

    def _update_consumption_timeline(self):
        for timestamp in range(len(self._storage_actions_timeline)):
            self._consumption[timestamp] = self._old_consumption[timestamp] + self._storage_actions_timeline[timestamp]

    def _update_storage_capacity_timeline(self):

        self._storage_capacity_timeline = _get_storage_capacity_timeline(self._storage_actions_timeline)
