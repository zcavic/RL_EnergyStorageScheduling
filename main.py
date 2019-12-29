import os
from environment.environment_discrete import EnvironmentDiscrete
import pandas as pd
from rl_algorithms.deep_q_learning import DeepQLearningAgent
from power_algorithms.heuristic_storage_scheduler import HeuristicStorageScheduler
import time

def load_dataset():
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, './dataset/data.csv')
    df = pd.read_csv(file_path)

    return df

def split_dataset(df, split_index):
    df_train = df[df.index <= split_index]
    df_test = df[df.index > split_index]

    return df_train, df_test

def main():
    #dataset contains power injection of nodes
    df = load_dataset()
    df_train, df_test = split_dataset(df, 998)

    print('=====================Heuristic calculation=====================')
    heuristic = HeuristicStorageScheduler() 
    heuristic.test(df_test)

    #environment should'n have the entire dataset as an input parameter, but train and test methods
    environment_discrete = EnvironmentDiscrete()

    print('=====================agent=====================')
    agent = DeepQLearningAgent(environment_discrete)

    n_episodes = 2
    print('agent training started')
    t1 = time.time()
    agent.train(df_train, n_episodes)
    t2 = time.time()
    print ('agent training finished in', t2-t1)

    agent.test(df_test)

if __name__ == '__main__':
    main()
