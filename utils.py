import os
from datetime import timedelta
from xml.dom import minidom
import matplotlib.pyplot as plt
import pandas as pd


def load_dataset():
    return load_dataset('./dataset/energy_storage_dataset.csv')


def load_dataset(path):
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, path)
    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df


def split_dataset(df, split_percentage):
    split_index = int(df.index.size / 24 * (1-split_percentage)) * 24 - 1
    split_datetime = df.index[0] + timedelta(hours=split_index)
    df_train = df[df.index <= split_datetime]
    df_test = df[df.index > split_datetime]
    return df_train, df_test


def split_dataset(df):
    df_test = select_random_day(df)
    df_train = df.drop(df_test.index)
    return df_train, df_test


def split_dataset_2(df):
    df_test = select_last_day(df)
    df_train = df.drop(df_test.index)
    return df_train, df_test


def extract_day_starts(df):
    return df[(df.index.hour == 0)]


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


def select_last_day(df):
    df_day_starts = extract_day_starts(df)
    day_df = get_day_from_day_start(df_day_starts.iloc[[-1]], df)
    return day_df


def select_random_day_start(df):
    return df[(df.index.hour == 0)].sample(n=1).index[0]


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


def plot_daily_results(day_id, proposed_storage_powers, actual_storage_powers, storage_socs, electricity_price):
    time = [i for i in range(24)]

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True)

    ax0.set_title('Electricity price')
    ax0.step(time, electricity_price, label='Electricity price')
    ax0.legend(loc='upper right')

    ax1.set_title('Storage Action')
    ax1.step(time, proposed_storage_powers, label='Proposed', color='k')
    ax1.step(time, actual_storage_powers, label='Actual', color='b')
    ax1.legend(loc='upper right')

    ax2.set_title('State of charge')
    ax2.plot(time, storage_socs, label='SOC', color='b')
    ax2.legend(loc='upper right')

    fig.savefig(str(day_id) + '_day_resuts.png')
    plt.show()


def _get_ddpg_conif(config_path):
    hidden_size = 128
    actor_learning_rate = 1e-5
    critic_learning_rate = 1e-4
    gamma = 1.0
    tau = 1e-3
    max_memory_size = 600000

    xml_doc = minidom.parse(config_path)
    alg_list = xml_doc.getElementsByTagName('Algorithm')
    for alg in alg_list:
        if alg.attributes['Name'].value == "DDPG":
            for config in alg.childNodes:
                if config.nodeName == 'hidden_size':
                    hidden_size = config.firstChild.nodeValue;
                if config.nodeName == 'actor_learning_rate':
                    actor_learning_rate = config.firstChild.nodeValue;
                if config.nodeName == 'critic_learning_rate':
                    critic_learning_rate = config.firstChild.nodeValue;
                if config.nodeName == 'gamma':
                    gamma = config.firstChild.nodeValue;
                if config.nodeName == 'tau':
                    tau = config.firstChild.nodeValue;
                if config.nodeName == 'max_memory_size':
                    max_memory_size = config.firstChild.nodeValue;

    return int(hidden_size), float(actor_learning_rate), float(critic_learning_rate), float(gamma), float(tau), int(max_memory_size)
