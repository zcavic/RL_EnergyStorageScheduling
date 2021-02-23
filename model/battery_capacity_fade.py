import math
import rainflow
import numpy as np
import matplotlib.pyplot as plt


def calculate(soc_series):
    if soc_series is None or len(soc_series) <= 1:
        return 0
    f_c = 0
    for rng, mean, count, i_start, i_end in rainflow.extract_cycles(soc_series):  # TODO da li je range isto sto i amplituda?
        soc_stress = _soc_stress(mean)
        dod_stress = _dod_stress(rng)
        f_c = f_c + (count * soc_stress * dod_stress)

    f_t = _time_stress(len(soc_series) * 3600) * _soc_stress(np.mean(soc_series))

    capacity_fade = _capacity_fade(f_c, f_t)
    return capacity_fade


def _soc_stress(mean):
    k_sigma = 1.04
    sigma_ref = 0.5
    return math.exp(k_sigma * (mean - sigma_ref))


def _dod_stress(rng):
    k_delta_1 = 1.40 * 10 ** 5
    k_delta_2 = -5.01 * 10 ** -1
    k_delta_3 = -1.23 * 10 ** 5
    return (k_delta_1 * rng ** k_delta_2 + k_delta_3) ** -1  # TODO proveriti da li treba 2, lici da ne treba


def _time_stress(sec):
    k_t = 4.14 * 10 ** -10
    return k_t * sec


def _capacity_fade(f_c, f_t):
    alfa = 5.75 * 10 ** -2
    beta = 121
    return 1 - alfa * math.exp(-beta * (f_c + f_t)) - (1 - alfa) * math.exp(-(f_c + f_t))


def test_calculation(soc_series):
    test = calculate(soc_series)
    capacity_fade = []
    for i in range(len(soc_series)):
        if i == 0:
            capacity_fade.append(0)
        elif i % 24 == 0:
            capacity_fade.append(calculate(soc_series[:i]))

    x_axis = [1 + j for j in range(len(capacity_fade))]
    plt.plot(x_axis, capacity_fade)
    plt.xlabel('time [days]')
    plt.ylabel('capacity fade [%}')
    plt.savefig("capacity_fade.png")
    plt.show()
