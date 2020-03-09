class ScheduleData(object):

    def __init__(self, capacity_timeline: list, action_timeline: list, consumption_curve: list):
        self.capacity_timeline = capacity_timeline
        self.action_timeline = action_timeline
        self.demand_forecast = consumption_curve
        self.availability_timeline = [1] * len(consumption_curve)

    def __copy__(self):
        return ScheduleData(self.capacity_timeline.copy(), self.action_timeline.copy(), self.demand_forecast.copy())

    def get_index_of_minimum_consumption(self):
        index = -1
        for i in range(len(self.demand_forecast)):
            if self.availability_timeline[i] == 0:
                continue
            elif index == -1:
                index = i
            elif self.demand_forecast[i] < self.demand_forecast[index]:
                index = i
        return index

    def get_index_of_maximum_consumption(self):
        index = -1
        for i in range(len(self.demand_forecast)):
            if self.availability_timeline[i] == 0:
                continue
            elif index == -1:
                index = i
            elif self.demand_forecast[i] > self.demand_forecast[index]:
                index = i
        return index

    def update_storage_capacity_timeline(self):
        capacity_timeline = [0] * len(self.action_timeline)
        for timestamp in range(len(self.action_timeline)):
            next_timestamp = timestamp
            while next_timestamp < len(self.action_timeline):
                capacity_timeline[next_timestamp] = capacity_timeline[next_timestamp] + \
                                                    self.action_timeline[timestamp]
                next_timestamp = next_timestamp + 1

        minimum = min(capacity_timeline)
        if minimum < 0:
            for timestamp in range(len(capacity_timeline)):
                capacity_timeline[timestamp] = capacity_timeline[timestamp] + \
                                               abs(minimum)

        self.capacity_timeline = capacity_timeline

    def update_consumption_timeline(self, consumption_curve: list):
        for timestamp in range(len(self.action_timeline)):
            self.demand_forecast[timestamp] = consumption_curve[timestamp] + self.action_timeline[timestamp]

    def is_valid_capacity_timeline(self, max_storage_capacity):
        if max(self.capacity_timeline) <= max_storage_capacity:
            return True
        else:
            return False

    def is_valid_action_timeline(self, max_storage_power):
        if max(self.action_timeline) < max_storage_power and abs(min(self.action_timeline)) < max_storage_power:
            return True
        else:
            return False

    def reset_availability(self):
        self.availability_timeline = [1] * len(self.demand_forecast)
