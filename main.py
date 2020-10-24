from dataset.dataset_helper_storage import create_dataset_scv
from environment.environment_continous import EnvironmentContinous
from environment.environment_discrete import EnvironmentDiscrete
from heuristic_algorithm.heuristic_storage_scheduler import HeuristicStorageScheduler
from regresion.train_regression_model import train_regression_model
from rl_algorithms.ddpg import DDPGAgent
from rl_algorithms.deep_q_learning import DeepQLearningAgent
import time
from utils import load_dataset, split_dataset


def main():
    create_dataset_scv()

if __name__ == '__main__':
    main()
