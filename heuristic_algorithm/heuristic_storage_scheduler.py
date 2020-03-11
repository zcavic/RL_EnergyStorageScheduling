from environment.energy_storage import EnergyStorage
from heuristic_algorithm.environment_heuristic import EnvironmentHeuristic
from heuristic_algorithm.schedule_data import ScheduleData


class HeuristicStorageScheduler(object):

    def __init__(self, environment_heuristic: EnvironmentHeuristic):
        self._energy_storage_collection = environment_heuristic.energy_storage_collection
        self.demand_forecast = environment_heuristic.demand_forecast
        self._step = 0.1  # this is step (percent of nominal power) for gradient charge or discharge of energy storage

    def start(self):
        schedule_data_collection = dict()
        for energy_storage in self._energy_storage_collection:
            schedule_data_collection[energy_storage.id] = self.calculate_storage_schedule(energy_storage)
            print(schedule_data_collection[energy_storage.id].action_timeline)
            for timestamp in range(len(self.demand_forecast)):
                energy_storage.send_action(schedule_data_collection[energy_storage.id].action_timeline[timestamp])

    def calculate_storage_schedule(self, energy_storage):
        schedule_data = ScheduleData([0] * len(self.demand_forecast), [0] * len(self.demand_forecast),
                                     self.demand_forecast.copy())
        charging = True
        discharging = True
        iteration = 0

        while charging and discharging and iteration < 1000:
            schedule_data_draft = schedule_data.__copy__()
            charging = self._charge(schedule_data_draft, energy_storage)
            schedule_data_draft.reset_availability()
            discharging = self._discharge(schedule_data_draft, energy_storage)
            schedule_data_draft.reset_availability()

            if not charging or not discharging:
                break

            schedule_data_draft.update_consumption_timeline(self.demand_forecast)
            if not schedule_data_draft.is_valid_action_timeline(energy_storage.max_p_kw):
                break
            if not schedule_data_draft.is_valid_capacity_timeline(energy_storage.max_e_mwh):
                break

            temp = 0
            for action in schedule_data_draft.action_timeline:
                temp = temp + action
            if temp > 0.0001:
                print('Error in action timeline!!!!!')

            schedule_data = schedule_data_draft
            iteration = iteration + 1

        return schedule_data

    def _charge(self, schedule_data: ScheduleData, energy_storage: EnergyStorage):

        timestamp = schedule_data.get_index_of_minimum_consumption()

        if timestamp == -1:
            return False

        if schedule_data.action_timeline[timestamp] >= 0:
            new_capacity = schedule_data.capacity_timeline[timestamp] + (energy_storage.max_p_kw * self._step)
            new_action = schedule_data.action_timeline[timestamp] + (energy_storage.max_p_kw * self._step)
            if new_capacity <= energy_storage.max_e_mwh and new_action <= energy_storage.max_p_kw:
                schedule_data.action_timeline[timestamp] = new_action
                schedule_data.update_storage_capacity_timeline()
                return True

        schedule_data.availability_timeline[timestamp] = 0

        return self._charge(schedule_data, energy_storage)

    def _discharge(self, schedule_data: ScheduleData, energy_storage: EnergyStorage):

        timestamp = schedule_data.get_index_of_maximum_consumption()

        if timestamp == -1:
            return False

        if schedule_data.action_timeline[timestamp] <= 0:
            new_capacity = schedule_data.capacity_timeline[timestamp] - (energy_storage.max_p_kw * self._step)
            new_action = schedule_data.action_timeline[timestamp] - (energy_storage.max_p_kw * self._step)
            if new_capacity >= 0 and abs(new_action) <= energy_storage.max_p_kw:
                schedule_data.action_timeline[timestamp] = new_action
                return True

        schedule_data.availability_timeline[timestamp] = 0

        return self._discharge(schedule_data, energy_storage)