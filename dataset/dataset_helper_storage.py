from datetime import datetime, timedelta

from model.battery_capacity_fade import calculate
from obsolete.environment.energy_storage import EnergyStorage
import obsolete.power_algorithms.network_management as nm
from obsolete.power_algorithms.power_flow import PowerFlow
import pandas as pd
import os

from utils import load_dataset, save_dataset


def create_dataset():
    _start_datetime = datetime(2000, 1, 1)
    _end_datetime = datetime(2001, 12, 31)
    _consumption_load_diagram = [0.2, 0.2, 0.1, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 0.9, 0.7, 0.6, 0.5, 0.5, 0.6, 0.7, 0.8, 1,
                                 1,
                                 0.8, 0.6, 0.5, 0.4, 0.3]
    _storage_load_diagram = [1, 1, 1, 1, 1, 1, 1, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0.2, 1, 1]
    # _storage_load_diagram = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0.2, 1]
    # _storage_load_diagram = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
    _electricity_price = [1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 10, 10, 10, 10, 10, 10, 10, 10]
    _solar_production = [0, 0, 0, 0, 0, 0, 0.5, 0.8, 1, 1, 1, 1, 1, 1, 1, 1, 0.8, 0.5, 0, 0, 0, 0, 0, 0]
    _energy_storage = _create_energy_storage()
    _df = pd.DataFrame(
        columns=['Date time', 'Electricity price', 'Solar Production', 'Consumption Load', 'Storage Load',
                 'State of charge', 'Days in idle', 'Number of cycles', 'Capacity fade'])
    index = 0
    sum_fade = 0
    for single_datetime in _date_range(_start_datetime, _end_datetime, delta=timedelta(hours=1)):
        _df.loc[index] = [single_datetime,
                          _electricity_price[single_datetime.hour],
                          _solar_production[single_datetime.hour],
                          _consumption_load_diagram[single_datetime.hour],
                          _energy_storage.energyStorageState.power,
                          _energy_storage.energyStorageState.soc / _energy_storage.max_e_mwh,
                          _energy_storage.energyStorageState.days_in_idle,
                          _energy_storage.energyStorageState.no_of_cycles,
                          (1 - _energy_storage.energyStorageState.fade)]
        _storage_load = _energy_storage.max_p_kw * _storage_load_diagram[single_datetime.hour]
        x, y, t = _energy_storage.send_action(_storage_load)
        index = index + 1
        if single_datetime.month == 1 and single_datetime.day == 1 and single_datetime.hour == 0:
            print(single_datetime)
        # if single_datetime.hour != 23:
        #     sum_fade += t
        # else:
        #     print('hour: ', single_datetime.hour, 'capacity fade: ', sum_fade * 10000)
        #     sum_fade = 0

    _save_energy_storage_df_to_csv(_df)


def add_capacity_fade_to_dataset(dataset_filepath):
    dataset = load_dataset(dataset_filepath)
    for ind in dataset.index:
        dataset.loc[ind, 'fade'] = calculate(dataset['SOC'][:ind].to_list())
        if ind.hour == 0 and ind.day == 1:
            print(ind)
    save_dataset(dataset, dataset_filepath)


def _date_range(start_date, end_date, delta=timedelta(days=1)):
    current_date = start_date
    while current_date < end_date:
        yield current_date
        current_date += delta


def _create_energy_storage():
    network_manager = nm.NetworkManagement()
    power_flow = PowerFlow(network_manager)
    return EnergyStorage(1, network_manager.power_grid, power_flow)


def _save_energy_storage_df_to_csv(df, path):
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, path)
    df.to_csv(file_path, index=True)
