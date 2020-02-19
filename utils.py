import os
import pandas as pd 
import datetime as dt
import matplotlib.pyplot as plt
from math import ceil
import numpy as np 
import gym

def load_dataset():
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, './dataset/data.csv')
    df = pd.read_csv(file_path, index_col=0)

    return df

def split_dataset(df, split_index):
    df_train = df[df.time <= split_index]
    df_test = df[df.time > split_index]

    return df_train, df_test

def extract_day_starts(df):
    return df[(df.time % 24 == 0)]
    
def get_day_from_day_start(day_start, df):
    ind = day_start.index.values[0]
    day_df = df[(df.index >= ind) & (df.index < ind + 24)]
    return day_df

def select_random_day(df):
    df_day_starts = extract_day_starts(df)
    #print(df_day_starts.describe())
    #print("df_day_starts")
    #print(df_day_starts)
    day_start_sample = df_day_starts.sample(n=1)
    #print("day_start_sample")
    #print(day_start_sample)
    day_df = get_day_from_day_start(day_start_sample, df)
    return day_df

def get_scaling_from_row(row):
    solar_columnes = ['solar1'] #todo hardcode da li ovo nekako povezati sa create dataset, a da ovo bude globalna promjenljiva u utils
    load_columnes = ['load1']
    solar_percents = row[solar_columnes].values.tolist() #list of scaling factors for every solar in the network
    load_percents = row[load_columnes].values.tolist() 
    return solar_percents, load_percents

def plot_daily_results(day_id, solar_powers, load_powers, storage_powers):
    time = [i for i in range(24)]

    plt.plot(time, solar_powers, label='Solar', color='g')
    plt.plot(time, load_powers, label='Load', color='r')
    plt.plot(time, storage_powers, label='Storage', color='k')
    plt.xlabel('Time') 
    plt.ylabel('Active power') 
    plt.legend(loc='upper right')
    plt.savefig(str(day_id) + '_day_resuts.png')
    plt.show()