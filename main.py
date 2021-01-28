from _New.ddpg_lite import DDPGAgentLite
from _New.energy_storage_lite import EnergyStorageLite, ChargingState, _test_energy_storage1, _test_energy_storage_2, \
    _test_energy_storage_3, _test_energy_storage_4
from _New.environment_ddpg import EnvironmentDDPG
import time
from utils import load_dataset, split_dataset


def main():
    # only when want to change something
    # create_dataset()
    # _test_energy_storage1()
    # _test_energy_storage_2()
    # _test_energy_storage_3()
    # _test_energy_storage_4()

    # Environment
    environment = EnvironmentDDPG()

    # Agent
    agent = DDPGAgentLite(environment)

    _start_agent(agent)


def _start_agent(agent):
    # Dataset
    df = load_dataset()
    df_train, df_test = split_dataset(df, 0.9)
    n_episodes = 10000
    print('agent training started')
    t1 = time.time()
    agent.train(n_episodes)
    t2 = time.time()
    print('agent training finished in', t2 - t1)
    agent.test()


if __name__ == '__main__':
    main()
