from _New.energy_storage_lite import EnergyStorageLite


def create_energy_storage():
    return EnergyStorageLite(max_p_mw=1, max_e_mwh=5, initial_soc=0.5)


def create_energy_storage_from_dataset(dataset_row):
    return EnergyStorageLite(max_p_mw=1, max_e_mwh=5, initial_soc=0.5)
