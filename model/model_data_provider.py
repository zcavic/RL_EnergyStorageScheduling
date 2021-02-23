from datetime import timedelta
import numpy as np
from model.battery_capacity_fade import calculate
from model.energy_storage import EnergyStorage


class ModelDataProvider:

    def __init__(self, dataset):
        self._dataset = dataset

    def get_electricity_price_for(self, timestamp):
        return self._dataset.loc[timestamp, 'price_day_ahead']

    def get_electricity_price_for_day(self, timestamp):
        end_day = timestamp + timedelta(hours=23)
        return self._dataset.loc[timestamp:end_day, 'price_day_ahead'].values

    def create_energy_storage(self, datetime):
        end_day = datetime-timedelta(hours=1)
        capacity_fade = calculate(np.array(self._dataset.loc[:end_day, 'SOC']))
        soc = self._dataset.loc[datetime, 'SOC']
        return EnergyStorage(max_p_mw=1, max_e_mwh=6, initial_soc=soc, capacity_fade=capacity_fade)
