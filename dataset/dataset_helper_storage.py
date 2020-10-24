from datetime import datetime, timedelta
from environment.energy_storage import EnergyStorage
import power_algorithms.network_management as nm
from power_algorithms.power_flow import PowerFlow
import pandas as pd
import os


def create_dataset_scv():
    _start_datetime = datetime(2000, 1, 1)
    _end_datetime = datetime(2003, 12, 31)
    _demand_load_diagram = [0.2, 0.2, 0.1, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 0.9, 0.7, 0.6, 0.5, 0.5, 0.6, 0.7, 0.8, 1, 1,
                            0.8, 0.6, 0.5, 0.4, 0.3]
    #_storage_load_diagram = [1, 1, 1, 1, 1, 1, 1, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0.2, 1, 1]
    _storage_load_diagram = [2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0.2, 1]
    _energy_storage = _create_energy_storage()
    _df = pd.DataFrame(columns=['Demand Load', 'Storage Load', 'State of charge', 'Capacity fade idle', 'Capacity fade Cycling',
                                'Capacity fade'])

    for single_datetime in _date_range(_start_datetime, _end_datetime, delta=timedelta(hours=1)):
        _storage_load = _energy_storage.max_p_kw * _storage_load_diagram[single_datetime.hour]
        actual_action, cant_execute = _energy_storage.send_action(_storage_load, single_datetime.hour)
        _df.loc[single_datetime] = [_demand_load_diagram[single_datetime.hour],
                                    actual_action,
                                    _energy_storage.energyStorageState.soc/_energy_storage.max_e_mwh,
                                    _energy_storage.energyStorageState.get_idle_fade(),
                                    _energy_storage.energyStorageState.get_cycle_fade(),
                                    (1 - (_energy_storage.energyStorageState.get_idle_fade()
                                          + _energy_storage.energyStorageState.get_cycle_fade()))]
        if single_datetime.month == 1 and single_datetime.day == 1 and single_datetime.hour == 1:
            print(single_datetime)

    _save_energy_storage_df_to_csv(_df)


def _date_range(start_date, end_date, delta=timedelta(days=1)):
    current_date = start_date
    while current_date < end_date:
        yield current_date
        current_date += delta


def _create_energy_storage():
    network_manager = nm.NetworkManagement()
    power_flow = PowerFlow(network_manager)
    return EnergyStorage(0, network_manager.power_grid, power_flow)


def _save_energy_storage_df_to_csv(df):
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, './energy_storage_dataset.csv')

    df.to_csv(file_path, index=True)
