import pandapower as pp
import power_algorithms.network_definition as grid
import pandas as pd
import os


def update_measurements():
    power_network = grid.create_cigre_network_mv()
    df = _get_dataframe()

    for timestamp in df.index:
        # set scaling
        for index in power_network.load.index:
            if power_network.load.name.loc[index] in df.columns:
                power_network.load.scaling.loc[index] = df[power_network.load.name.loc[index]][timestamp]
            else:
                power_network.load.scaling.loc[index] = 0
        for index in power_network.sgen.index:
            if power_network.sgen.name.loc[index] in df.columns:
                power_network.sgen.scaling.loc[index] = df[power_network.sgen.name.loc[index]][timestamp]
            else:
                power_network.sgen.scaling.loc[index] = 0

        # calculate LF
        _calculate_power_flow(power_network)

        # get LF results and add to measurements
        for index in power_network.bus.index:
            if power_network.bus.name.loc[index] in df.columns:
                df[power_network.bus.name.loc[index]][timestamp] = power_network.res_bus.p_mw.loc[0]

        for index in power_network.line.index:
            if power_network.line.name.loc[index] in df.columns:
                df[power_network.line.name.loc[index]][timestamp] = power_network.res_line.p_from_kw.l.loc[0]

        for index in power_network.storage.index:
            if power_network.storage.name.loc[index] in df.columns:
                df[power_network.storage.name.loc[index]][timestamp] = power_network.storage.p_kw.l.loc[0]

    _save_dataframe_to_csv(df)


def _get_dataframe():
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, './cigre_dataset.csv')

    df = pd.read_csv(file_path, index_col=0)

    return df


def _save_dataframe_to_csv(df):
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, './cigre_dataset_all.csv')

    df.to_csv(file_path, index=True)


def _calculate_power_flow(power_network):
    pp.runpp(power_network, algorithm="bfsw", calculate_voltage_angles=False)
