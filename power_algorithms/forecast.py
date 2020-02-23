class Forecast(object):

    def __init__(self):
        self.load = [0.2, 0.2, 0.1, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 0.9, 0.7, 0.6, 0.5, 0.5, 0.6, 0.7, 0.8, 1, 1, 0.8, 0.6, 0.5, 0.4, 0.3]
        self.production = [0, 0, 0, 0, 0, 0.5, 0.8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.8, 0.5, 0, 0, 0, 0]

        self.nominal_load = 100
        self.nominal_production = 50
        self.consumption = [None] * len(self.load)
        self._calculate_consumption()

    def _calculate_consumption(self):
        for i in range(len(self.load)):
            self.consumption[i] = self.load[i] * self.nominal_load - self.production[i] * self.nominal_production