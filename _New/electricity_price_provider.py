price = [10, 10, 10, 10, 40, 40, 40, 80, 80, 80, 80, 80, 80, 80, 40, 40, 40, 40, 40, 40, 10, 10, 10, 10]


def get_electricity_price_for(time_step):
    return price[time_step-1]


def get_electricity_price():
    return price
