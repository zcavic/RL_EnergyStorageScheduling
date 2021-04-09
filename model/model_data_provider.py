from datetime import timedelta
import numpy as np
from model.energy_storage import EnergyStorage


class ModelDataProvider:

    def __init__(self, dataset):
        self._dataset = dataset

    def get_electricity_price(self, timestamp):
        return self._dataset.loc[timestamp, 'price_day_ahead']

    def get_capacity_fade(self, timestamp):
        return self._dataset.loc[timestamp, 'capacity_fade']

    def get_electricity_price_for_day(self, timestamp):
        end_day = timestamp + timedelta(hours=23)
        return self._dataset.loc[timestamp:end_day, 'price_day_ahead'].values

    def get_soc_history(self, timestamp):
        return np.array(self._dataset.loc[:timestamp, 'SOC'])

    def create_energy_storage(self, datetime):
        end_day = datetime-timedelta(hours=1)
        soc = self._dataset.loc[datetime, 'SOC']
        return EnergyStorage(max_p_mw=1, max_e_mwh=6, initial_soc=soc, soc_history=self.get_soc_history(end_day))
