from utils import load_dataset


class ModelDataProvider:

    def __init__(self, dataset_path):
        self._df = load_dataset(dataset_path)

    def get_electricity_price_for(self, time_step):
        return self._df['ElectricityPrice'].values[time_step - 1]

    def get_electricity_price(self):
        return self._df['ElectricityPrice'].values

