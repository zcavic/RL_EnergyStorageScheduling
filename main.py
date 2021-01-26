from _New.ddpg_lite import DDPGAgentLite
from _New.environment_ddpg import EnvironmentDDPG
import time
from utils import load_dataset, split_dataset


def main():
    # only when want to change something
    # create_dataset()

    # Environment
    environment = EnvironmentDDPG()

    # Agent
    agent = DDPGAgentLite(environment)

    _start_agent(agent)


def _start_agent(agent):
    # Dataset
    df = load_dataset()
    df_train, df_test = split_dataset(df, 0.9)
    n_episodes = 1000
    print('agent training started')
    t1 = time.time()
    agent.train(df_train, n_episodes)
    t2 = time.time()
    print('agent training finished in', t2 - t1)
    agent.test(df_test)


if __name__ == '__main__':
    main()
