import power_algorithms.network_definition as grid
import pandapower as pp
import pandas as pd

class NetworkManagement:
    def __init__(self):
         self.power_grid = grid.create_network()

    def get_power_grid(self):
        return self.power_grid
    
    # For given capacitor switch name (CapSwitch1, CapSwitch2...) status is changed.
    def change_capacitor_status(self, capSwitchName, closed):
        switchIndex = pp.get_element_index(self.power_grid, "switch", capSwitchName)
        self.power_grid.switch.closed.loc[switchIndex] = closed
    
    def toogle_capacitor_status(self, capSwitchName):
        switchIndex = pp.get_element_index(self.power_grid, "switch", capSwitchName)
        currentState = self.power_grid.switch.closed.loc[switchIndex]
        self.power_grid.switch.closed.loc[switchIndex] = not currentState

    def get_all_capacitor_switch_names(self):
        return self.power_grid.switch['name'].tolist()

    def get_all_capacitors(self):
        return pd.Series(self.power_grid.switch.closed.values, index=self.power_grid.switch.name).to_dict()

    def set_load_scaling(self, scaling_factors):
        if (len(scaling_factors) != len(self.power_grid.load.index)):
            print("(ERROR) Input list of scaling factors {} is not the same length as number of loads {}".format(len(scaling_factors), len(self.power_grid.load.index)))
            return

        for index, load in self.power_grid.load.iterrows():
            self.power_grid.load.scaling.loc[index] = scaling_factors[index]

    def set_capacitors_initial_status(self, capacitors_statuses):
        capacitor_indices = self.get_capacitor_indices_from_shunts()
        if (len(capacitors_statuses) != len(capacitor_indices)):
            print("(ERROR) Input list of capacitor statuses {} is not the same length as number of capacitors {}".format(len(capacitors_statuses), len(self.power_grid.shunt.index)))
            return
        
        capacitor_switches = self.power_grid.switch.index.tolist()
        input_status_index = 0
        for switch_index in capacitor_switches:
            self.power_grid.switch.closed.loc[switch_index] = capacitors_statuses[input_status_index]
            input_status_index += 1

    def get_capacitor_indices_from_shunts(self):
        capacitors = []
        for index, row in self.power_grid.shunt.iterrows():
            if 'Cap' in row['name']:
                capacitors.append(index)
        return capacitors

    def print_cap_status(self):
        print(self.power_grid.switch)