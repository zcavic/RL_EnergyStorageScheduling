import pickle
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from utils import load_dataset

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


def train_regression_model():
    df = load_dataset()

    x = _get_x(df)
    y = _get_y(df)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=28)

    # RandomForestRegressor
    clf = RandomForestRegressor(random_state=1)
    print('training regression started')
    t1 = time.time()
    clf.fit(x_train, y_train)
    t2 = time.time()
    print('training regression finished in', t2 - t1)
    y_pred = clf.predict(x_test)

    pickle.dump(clf, open('Aguas_Santas_regression_model_with_outdoor.sav', 'wb'))

    fig2 = plt.figure(figsize=(6, 6))
    plt.plot(y_test, y_test, c='k')
    plt.scatter(y_test, y_pred, c='g')
    plt.xlabel('Real Measurements')
    plt.ylabel('Predicted Measurements')
    plt.title("Real Measurements vs Predicted Measurements")
    fig2.savefig('regression_1.png')


def _get_x(df):
    x = df[['Bus 1', 'Bus 12', 'Bus 4', 'Bus 9', 'Bus 7', 'Bus 14', 'Battery 1', 'Battery 2']]
    x.loc[:, 'Battery 1 Command'] = df.loc[:, 'Battery 1'].shift(-1)
    x.loc[:, 'Battery 2 Command'] = df.loc[:, 'Battery 2'].shift(-1)
    return x.iloc[1:].iloc[:-1]


def _get_y(df):
    y = df[['Bus 1', 'Bus 12', 'Bus 4', 'Bus 9', 'Bus 7', 'Bus 14', 'Battery 1', 'Battery 2']].shift(1)
    return y.iloc[1:].iloc[:-1]
