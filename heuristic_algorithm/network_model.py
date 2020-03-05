from environment.energy_storage import EnergyStorage
from power_algorithms.network_management import NetworkManagement
from power_algorithms.power_flow import PowerFlow


class NetworkModel:
    def __init__(self):
        self.network_manager = NetworkManagement()
        self.power_flow = PowerFlow(self.network_manager)
        self.energy_storage = EnergyStorage(1, 10, 10, 0)
        self._calculate_consumption()

    def _calculate_consumption(self):
        self.load = [0.2, 0.2, 0.1, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 0.9, 0.7, 0.6, 0.5, 0.5, 0.6, 0.7, 0.8, 1, 1, 0.8,
                     0.6, 0.5, 0.4, 0.3]
        self.production = [0, 0, 0, 0, 0, 0.5, 0.8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.8, 0.5, 0, 0, 0, 0]
        self.nominal_load = 100
        self.nominal_production = 50
        self.consumption = [None] * len(self.load)
        self._calculate_consumption()
        for i in range(len(self.load)):
            self.consumption[i] = self.load[i] * self.nominal_load - self.production[i] * self.nominal_production

    def send_action(self, action):
        if action < 0:
            self.energy_storage.discharge(action)
        elif action > 0:
            self.energy_storage.charge(action)
        else:
            self.energy_storage.turn_off()

    def _update_model(self):
        self.network_manager.set_storage_scaling(self.energy_storage.get_power)
        self.power_flow.calculate_power_flow()