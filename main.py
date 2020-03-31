#from environment.environment_continous import EnvironmentContinous
from environment.environment_discrete import EnvironmentDiscrete
from heuristic_algorithm.heuristic_storage_scheduler import HeuristicStorageScheduler
from rl_algorithms.ddpg import DDPGAgent
from rl_algorithms.deep_q_learning import DeepQLearningAgent
import time
from utils import load_dataset, split_dataset


def main():
    #dataset contains power injection of nodes
    df = load_dataset()
    df_train, df_test = split_dataset(df, 71)

    #print('==============Heuristic algorithm==============')
    #heuristic_algorithm = HeuristicStorageScheduler()
    #heuristic_algorithm.start()

    #network_manager = nm.NetworkManagement()
    #power_flow = PowerFlow(network_manager)
    #power_flow.create_data_set()

    #environment should'n have the entire dataset as an input parameter, but train and test methods
    environment_discrete = EnvironmentDiscrete()
    #environment_continous = EnvironmentContinous()

    print('=====================agent=====================')
    agent = DeepQLearningAgent(environment_discrete)
    #agent = DDPGAgent(environment_continous)

    n_episodes = 750
    print('agent training started')
    t1 = time.time()
    agent.train_with_weight_averaging(df_train, n_episodes)
    t2 = time.time()
    print ('agent training finished in', t2-t1)

    agent.test(df_test)

if __name__ == '__main__':
    main()
