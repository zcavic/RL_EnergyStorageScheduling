import pickle
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from utils import load_dataset
from sklearn import metrics
from sklearn.multioutput import MultiOutputRegressor

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


def train_regression_model():
    df = load_dataset()

    x = _get_x(df)
    y = _get_y(df)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=28)

    # RandomForestRegressor
    clf = MultiOutputRegressor(RandomForestRegressor(random_state=1))
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

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


def _get_x(df):
    x = df[['Trafo 0-1', 'Trafo 0-12', 'Line 3-8', 'Line 3-4', 'Line 9-10', 'Line 7-8', 'Line 13-14', 'Battery 1', 'Battery 2']]
    x.loc[:, 'Battery 1 Command'] = df.loc[:, 'Battery 1'].shift(-1)
    x.loc[:, 'Battery 2 Command'] = df.loc[:, 'Battery 2'].shift(-1)
    return x.iloc[1:].iloc[:-1]


def _get_y(df):
    y = df[['Trafo 0-1', 'Trafo 0-12', 'Line 3-8', 'Line 3-4', 'Line 9-10', 'Line 7-8', 'Line 13-14']].shift(1)
    return y.iloc[1:].iloc[:-1]
