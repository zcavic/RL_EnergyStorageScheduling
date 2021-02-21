import math
import rainflow
from pandas import np


def calculate(soc_series):
    f_c = 0
    for rng, mean, count, i_start, i_end in rainflow.extract_cycles(soc_series):
        soc_stress = _soc_stress(mean)
        dod_stress = _dod_stress(rng)
        time_stress = _time_stress((i_end - i_start) * 60 * 60)
        f_c = f_c + (count * soc_stress * dod_stress * time_stress)

    f_t = _time_stress(len(soc_series) * 3600) * _soc_stress(np.mean(soc_series))

    capacity_fade = _capacity_fade(f_c, f_t)
    print(capacity_fade)


def _soc_stress(mean):
    k_sigma = 1.04
    sigma_ref = 0.5
    return math.exp(k_sigma * (mean - sigma_ref))


def _dod_stress(rng):
    k_delta_1 = 1.40 * 10 ** 5
    k_delta_2 = -5.01 * 10 ** -1
    k_delta_3 = -1.23 * 10 ** 5
    return (k_delta_1 * (2 * rng) ** k_delta_2 + k_delta_3) ** -1


def _time_stress(sec):
    k_t = 4.14 * 10 ** -10
    return k_t * sec


def _capacity_fade(f_c, f_t):
    alfa = 5.75 * 10 ** -2
    beta = 121
    return 1 - alfa * math.exp(-beta * (f_c + f_t)) - (1 - alfa) * math.exp(-(f_c + f_t))