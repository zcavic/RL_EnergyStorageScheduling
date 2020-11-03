from dataset.dataset_helper_storage import create_dataset
from environment.environment_continous import EnvironmentContinous
from rl_algorithms.ddpg import DDPGAgent
import time
from utils import load_dataset, split_dataset


def main():
    # only when want to change something
    create_dataset()

    # dataset contains power injection of nodes
    df = load_dataset()
    df_train, df_test = split_dataset(df, 647)

    # environment should'n have the entire dataset as an input parameter, but train and test methods
    # environment_discrete = EnvironmentDiscrete()
    environment_continous = EnvironmentContinous()

    print('=====================agent=====================')
    # agent = DeepQLearningAgent(environment_discrete)
    agent = DDPGAgent(environment_continous)

    n_episodes = 10000
    print('agent training started')
    t1 = time.time()
    agent.train(df_train, n_episodes)
    t2 = time.time()
    print('agent training finished in', t2 - t1)

    agent.test(df_test)


if __name__ == '__main__':
    main()
