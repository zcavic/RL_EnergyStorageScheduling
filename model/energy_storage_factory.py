from model.energy_storage import EnergyStorage


def create_energy_storage():
    return EnergyStorage(max_p_mw=1, max_e_mwh=6, initial_soc=0)

