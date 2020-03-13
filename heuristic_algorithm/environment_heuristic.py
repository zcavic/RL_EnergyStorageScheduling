from environment.energy_storage import EnergyStorage
import power_algorithms.network_definition as grid
import power_algorithms.network_management as nm
from power_algorithms.power_flow import PowerFlow


class EnvironmentHeuristic:
    def __init__(self):
        self.network_manager = nm.NetworkManagement()
        self.power_flow = PowerFlow(self.network_manager)
        self.power_grid = self.network_manager.power_grid
        self.energy_storage_collection = self._create_energy_storage()
        self.demand_forecast = self._get_demand_forecast()

    def set_scaling(self, timestamp):
        for index, load in self.power_grid.load.iterrows():
            self.power_grid.load.scaling.loc[index] = self._get_load_curve()[timestamp]
        for index, sgen in self.power_grid.sgen.iterrows():
            self.power_grid.sgen.scaling.loc[index] = self._get_production_curve()[timestamp]

    def _get_demand_forecast(self):
        consumption_forecast = self._get_consumption_forecast()
        production_forecast = self._get_production_forecast()
        demand_forecast = [0] * len(consumption_forecast)
        for i in range(len(consumption_forecast)):
            demand_forecast[i] = consumption_forecast[i] - production_forecast[i]
        return demand_forecast

    def _get_consumption_forecast(self):
        load_curve = self._get_load_curve()
        load = 0
        for index in self.power_grid.load.index:
            load = load + self.power_grid.load.loc[index].p_mw

        return [i * load for i in load_curve]

    def _get_production_forecast(self):
        production_curve = self._get_production_curve()
        production = 0
        for index in self.power_grid.sgen.index:
            production = production + self.power_grid.sgen.loc[index].p_mw

        return [i * production for i in production_curve]

    def _calculate_consumption(self):
        self.load = self._get_load_curve()
        self.production = self._get_production_curve()
        self.nominal_load = 100
        self.nominal_production = 50
        self.consumption = [None] * len(self.load)
        for i in range(len(self.load)):
            self.consumption[i] = self.load[i] * self.nominal_load - self.production[i] * self.nominal_production

    def _create_energy_storage(self):
        energy_storage_collection = []
        for index in self.power_grid.storage.index:
            energy_storage_collection.append(EnergyStorage(index, self.power_grid, self.power_flow))
        return energy_storage_collection

    def _get_load_curve(self):
        return [0.2, 0.2, 0.1, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 0.9, 0.7, 0.6, 0.5, 0.5, 0.6, 0.7, 0.8, 1, 1, 0.8,
                0.6, 0.5, 0.4, 0.3]

    def _get_production_curve(self):
        return [0, 0, 0, 0, 0, 0.5, 0.8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.8, 0.5, 0, 0, 0, 0]
