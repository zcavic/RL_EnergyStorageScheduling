from power_algorithms.power_flow import PowerFlow
import power_algorithms.network_management as nm

class HeuristicStorageScheduler(object):

    def __init__(self):
        self.network_manager = nm.NetworkManagement()
        self.power_flow = PowerFlow(self.network_manager)


    def calculate(self):
        pass


    #poziva calculate za svaki primjer iz df_test
    def test(self, df_test):
        self.power_flow.calculate_power_flow()
        print (self.power_flow.get_bus_voltages())
        print (self.power_flow.get_lines_apparent_power())