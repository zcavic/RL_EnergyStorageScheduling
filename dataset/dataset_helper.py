import pandapower as pp
import power_algorithms.network_definition as grid
import pandas as pd
import os
import random


def create_dataset_scv():
    power_network = grid.create_cigre_network_mv()
    df = _get_df()

    for timestamp in df.index:
        _update_network(power_network, df, timestamp)
        _calculate_power_flow(power_network)
        _update_df_with_measurements(power_network, df, timestamp)

    _save_df_to_csv(df)


def _update_network(power_network, df, timestamp):
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
    for index in power_network.storage.index:
        power_network.storage.scaling.loc[index] = random.random()*random.choice((-1, 1))


def _update_df_with_measurements(power_network, df, timestamp):
    for index in power_network.bus.index:
        if power_network.bus.name.loc[index] in df.columns:
            df[power_network.bus.name.loc[index]][timestamp] = power_network.res_bus.p_mw.loc[index]

    for index in power_network.line.index:
        if power_network.line.name.loc[index] in df.columns:
            df[power_network.line.name.loc[index]][timestamp] = power_network.res_line.p_from_mw.loc[index]

    for index in power_network.storage.index:
        if power_network.storage.name.loc[index] in df.columns:
            df[power_network.storage.name.loc[index]][timestamp] = power_network.res_storage.p_mw.loc[index]


def _get_df():
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, './cigre_dataset.csv')

    df = pd.read_csv(file_path, index_col=0)

    return df


def _save_df_to_csv(df):
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, './cigre_dataset_all.csv')

    df.to_csv(file_path, index=True)


def _calculate_power_flow(power_network):
    pp.runpp(power_network, algorithm="bfsw", calculate_voltage_angles=False)
