import os
import matplotlib.pyplot as plt
import pandas as pd

from environment.energy_storage import ChargingState, DischargingState, IdleState


def load_dataset():
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, './dataset/energy_storage_dataset.csv')
    df = pd.read_csv(file_path, index_col=0)
    return df


def split_dataset(df, split_index):
    df_train = df[df.index <= split_index]
    df_test = df[df.index > split_index]

    return df_train, df_test


def extract_day_starts(df):
    return df[(df.index % 24 == 0)]


def get_day_from_day_start(day_start, df):
    ind = day_start.index.values[0]
    day_df = df[(df.index >= ind) & (df.index < ind + 24)]
    return day_df


def select_random_day(df):
    df_day_starts = extract_day_starts(df)
    # print(df_day_starts.describe())
    # print("df_day_starts")
    # print(df_day_starts)
    day_start_sample = df_day_starts.sample(n=1)
    # print("day_start_sample")
    # print(day_start_sample)
    day_df = get_day_from_day_start(day_start_sample, df)
    return day_df


# vraca vrijednosti svih solara i loadova za jedan trenutak
def get_scaling_from_row(row):
    solar_columnes = [
        'Solar Production']  # todo hardcode da li ovo nekako povezati sa create dataset, a da ovo bude globalna promjenljiva u utils
    load_columnes = ['Consumption Load']
    electricity_price_columnes = ['Electricity price']
    solar_percents = row[solar_columnes].values.tolist()  # list of scaling factors for every solar in the network
    load_percents = row[load_columnes].values.tolist()
    electricity_price = row[electricity_price_columnes].values.tolist()
    return solar_percents, load_percents, electricity_price


def get_energy_storage_state(row):
    state_of_charge_columns = ['State of charge']
    days_in_idle_columns = ['Days in idle']
    no_of_cycles_columns = ['Number of cycles']
    storage_load_columns = ['Storage Load']
    state_of_charge = row[state_of_charge_columns].values.tolist()[0]
    days_in_idle = row[days_in_idle_columns].values.tolist()[0]
    no_of_cycles = row[no_of_cycles_columns].values.tolist()[0]
    storage_load = row[storage_load_columns].values.tolist()[0]
    return state_of_charge, days_in_idle, no_of_cycles, storage_load


def plot_daily_results(day_id, solar_powers, load_powers, proposed_storage_powers, actual_storage_powers, storage_socs,
                       electricity_price):
    time = [i for i in range(24)]

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True)

    ax0.set_title('Powers')
    ax0.step(time, solar_powers, label='Solar', color='g')
    ax0.step(time, load_powers, label='Load', color='r')
    ax0.step(time, electricity_price, label='Electricity price')
    ax0.legend(loc='upper right')

    ax1.set_title('Powers Action')
    ax1.step(time, proposed_storage_powers, label='Storage proposed', color='k')
    ax1.step(time, actual_storage_powers, label='Storage actual', color='b')
    ax1.legend(loc='upper right')

    ax2.set_title('State of charge')
    ax2.plot(time, storage_socs, label='Storage soc', color='b')
    ax2.legend(loc='upper right')

    fig.savefig(str(day_id) + '_day_resuts.png')
    plt.show()
