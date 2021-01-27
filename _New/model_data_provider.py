price = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
consumption = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]


def get_electricity_price_for(time_step):
    return price[time_step-1]


def get_power_consumption_for(time_step):
    return consumption[time_step-1]


def get_electricity_price():
    return price
