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
    df_train = df[df.index <= split_index]
    df_test = df[df.index > split_index]

    return df_train, df_test

def extract_day_starts(df):
    return df[(df.index % 24 == 0)]

def select_random_day(df):
    df_day_starts = extract_day_starts(df)
    day_start_sample = df_day_starts.sample(n=1)
    ind = day_start_sample.index.values[0]
    day_df = df[(df.index >= ind) & (df.index < ind + 24)]
    return day_df

def get_scaling_from_row(row):
    solar_columnes = ['solar1'] #todo hardcode da li ovo nekako povezati sa create dataset, a da ovo bude globalna promjenljiva u utils
    load_columnes = ['load1']
    solar_percents = row[solar_columnes].values.tolist() #list of scaling factors for every solar in the network
    load_percents = row[load_columnes].values.tolist() 
    return solar_percents, load_percents