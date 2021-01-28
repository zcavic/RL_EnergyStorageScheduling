from _New.energy_storage_lite import EnergyStorageLite


def create_energy_storage():
    return EnergyStorageLite(max_p_mw=1, max_e_mwh=6, initial_soc=0)

