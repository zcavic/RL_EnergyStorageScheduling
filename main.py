import os
from environment.environment_discrete import EnvironmentDiscrete
import pandas as pd
from rl_algorithms.deep_q_learning import DeepQLearningAgent
from power_algorithms.heuristic_storage_scheduler import HeuristicStorageScheduler
from power_algorithms.power_flow import PowerFlow
import power_algorithms.network_management as nm
import time
from utils import load_dataset, split_dataset


def main():
    #dataset contains power injection of nodes
    df = load_dataset()
    df_train, df_test = split_dataset(df, 47)

    #network_manager = nm.NetworkManagement()
    #power_flow = PowerFlow(network_manager)
    #power_flow.create_data_set()

    #print('=====================Heuristic calculation=====================')
    #heuristic = HeuristicStorageScheduler() 
    #heuristic.test(df_test)

    #environment should'n have the entire dataset as an input parameter, but train and test methods
    environment_discrete = EnvironmentDiscrete()

    print('=====================agent=====================')
    agent = DeepQLearningAgent(environment_discrete)

    n_episodes = 500
    print('agent training started')
    t1 = time.time()
    agent.train(df_train, n_episodes)
    t2 = time.time()
    print ('agent training finished in', t2-t1)

    agent.test(df_test)

if __name__ == '__main__':
    main()
