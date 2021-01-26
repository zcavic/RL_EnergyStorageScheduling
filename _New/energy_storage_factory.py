from _New.energy_storage_lite import EnergyStorageLite


def create_energy_storage():
    return EnergyStorageLite(max_p_mw=0.8, max_e_mwh=7)


def create_energy_storage_from_dataset(dataset_row):
    return EnergyStorageLite(max_p_mw=0.8, max_e_mwh=7)
