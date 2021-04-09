from dataset.dataset_helper_storage import add_capacity_fade_to_dataset
from ddpg.ddpg import DDPGAgent
from ddpg.environment_ddpg import EnvironmentDDPG
import time
from utils import _get_ddpg_conif, load_dataset, split_dataset, split_dataset_2


def main():
    dataset = load_dataset('./dataset/dataset_test3.csv')
    df_train, df_test = split_dataset_2(dataset)
    agent = _create_agent(dataset)

    print('agent training started')
    t1 = time.time()
    agent.train(1000, df_train)
    t2 = time.time()
    print('agent training finished in', t2 - t1)
    agent.test(df_test)


def _create_agent(_df):
    environment = EnvironmentDDPG(_df)
    hidden_size, actor_learning_rate, critic_learning_rate, gamma, tau, max_memory_size = _get_ddpg_conif(
        'rl_config.xml')
    return DDPGAgent(environment, hidden_size, actor_learning_rate, critic_learning_rate, gamma, tau,
                     max_memory_size)


def _training_and_test_dataset(dataset_path):
    _df = load_dataset(dataset_path)
    return split_dataset(_df, 0.1)


if __name__ == '__main__':
    main()
