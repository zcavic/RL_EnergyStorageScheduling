import numpy as np
from _New.battery_capacity_fade import calculate
from _New.energy_storage_lite import EnergyStorageLite


class ModelDataProvider:

    def __init__(self, dataset):
        self._dataset = dataset

    def get_electricity_price_for(self, timestamp):
        return self._dataset.loc[timestamp, 'price_day_ahead']

    def get_electricity_price(self):
        return self._dataset['ElectricityPrice'].values

    def create_energy_storage(self, datetime):
        capacity_fade = calculate(np.array(self._dataset.loc[:datetime, 'SOC']))
        soc = self._dataset.loc[datetime, 'SOC']
        return EnergyStorageLite(max_p_mw=1, max_e_mwh=6, initial_soc=soc, capacity_fade=capacity_fade)
