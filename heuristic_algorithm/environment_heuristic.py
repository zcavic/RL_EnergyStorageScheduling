from environment.energy_storage import EnergyStorage
import power_algorithms.network_definition as grid
from power_algorithms.power_flow import PowerFlow


class EnvironmentHeuristic:
    def __init__(self):
        self.network_manager = grid.create_network()
        self.power_flow = PowerFlow(self.network_manager)
        self.energy_storage = self._create_energy_storage()
        self._calculate_consumption()

    def _calculate_consumption(self):
        self.load = [0.2, 0.2, 0.1, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 0.9, 0.7, 0.6, 0.5, 0.5, 0.6, 0.7, 0.8, 1, 1, 0.8,
                     0.6, 0.5, 0.4, 0.3]
        self.production = [0, 0, 0, 0, 0, 0.5, 0.8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.8, 0.5, 0, 0, 0, 0]
        self.nominal_load = 100
        self.nominal_production = 50
        self.consumption = [None] * len(self.load)
        for i in range(len(self.load)):
            self.consumption[i] = self.load[i] * self.nominal_load - self.production[i] * self.nominal_production

    def _create_energy_storage(self):
        energy_storage_collection = []
        for index in self.power_grid.storage.index:
            energy_storage_collection.append(EnergyStorage(index, self.power_grid))
        return energy_storage_collection
