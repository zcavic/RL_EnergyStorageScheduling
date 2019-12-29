import pandapower as pp
import power_algorithms.network_management as nm
import math
import random
import pandas

class PowerFlow:
    def __init__(self, grid_creator):
        self.power_grid = grid_creator.get_power_grid()
        self.network_manager = grid_creator

    def calculate_power_flow(self):
        pp.runpp(self.power_grid, algorithm="bfsw", calculate_voltage_angles=False)

    def get_losses(self):
        grid_losses = 0
        for line_losses in self.power_grid.res_line.pl_mw:
            grid_losses += line_losses
        for transformer_losses in self.power_grid.res_trafo.pl_mw:
            grid_losses += transformer_losses
        
        return grid_losses

    def get_bus_voltages(self):
        switch_busses = set(self.power_grid.switch.element.values)
        mv_buses = self.power_grid.bus[(self.power_grid.bus.vn_kv == 20) & (~self.power_grid.bus.index.isin(switch_busses))].index
        name_with_voltage = {}
        for bus_index in mv_buses:
            name_with_voltage.update( {self.power_grid.bus.name.at[bus_index] : self.power_grid.res_bus.vm_pu.at[bus_index]} )

        return name_with_voltage

    def get_network_injected_p(self):
        return self.power_grid.res_ext_grid.p_mw

    def get_network_injected_q(self):
        return self.power_grid.res_ext_grid.q_mvar

    def get_lines_apparent_power(self):
        line_name_with_apparent_power = {}
        for index, line in self.power_grid.res_line.iterrows():
            p = line['p_from_mw']
            q = line['q_from_mvar']
            s = math.sqrt(pow(p, 2) + pow(q, 2))
            line_name_with_apparent_power.update( {self.power_grid.line.name.at[index] : s} )

        return line_name_with_apparent_power

    def get_capacitor_calculated_q(self):
        capacitor_q_injected = {}
        capacitors = self.network_manager.get_capacitor_indices_from_shunts()
        for cap_index in capacitors:
            capacitor_q_injected.update( {self.power_grid.shunt.name.at[cap_index] : self.power_grid.res_shunt.q_mvar.at[cap_index]} )

        return capacitor_q_injected
    
    def create_data_set(self):
        n_capacitors = len(self.network_manager.get_all_capacitors())
        n_consumers = len(self.network_manager.power_grid.load.index)

        columns = [i for i in range(n_consumers + n_capacitors)]
        index = [i for i in range(1000)]
        df = pandas.DataFrame(index=index, columns=columns)
        df = df.fillna(0)
        for index, row in df.iterrows():
            for i in range(n_consumers):
                df.loc[index, i] = random.random()
            for i in range(n_consumers, n_consumers + n_capacitors):
                df.loc[index, i] = random.choice([True, False])

        df.to_csv('data.csv')
