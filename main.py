from _New.ddpg_lite import DDPGAgentLite
from _New.environment_ddpg import EnvironmentDDPG
import time
from utils import _get_ddpg_conif


def main():
    n_episodes = 1000
    environment = EnvironmentDDPG('./dataset/Test1.csv')

    hidden_size, actor_learning_rate, critic_learning_rate, gamma, tau, max_memory_size = _get_ddpg_conif('rl_config.xml')
    agent = DDPGAgentLite(environment, hidden_size, actor_learning_rate, critic_learning_rate, gamma, tau, max_memory_size)
    _start_agent(agent, n_episodes)


def _start_agent(agent, n_episodes):
    print('agent training started')
    t1 = time.time()
    agent.train(n_episodes)
    t2 = time.time()
    print('agent training finished in', t2 - t1)
    agent.test()



if __name__ == '__main__':
    main()
