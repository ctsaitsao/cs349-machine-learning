from src import MultiArmedBandit, QLearning
import numpy as np
import matplotlib.pyplot as plt
import gym


def questions_2a_and_2b():
    env1 = gym.make('SlotMachines-v0')
    bandit_rewards_list = []

    for _ in range(10):
        bandit = MultiArmedBandit()
        _, rewards = bandit.fit(env1, steps=100000)
        bandit_rewards_list.append(rewards)

    plt.figure()
    plt.plot(range(100), bandit_rewards_list[0], label="first trial")
    plt.plot(range(100), sum(bandit_rewards_list[0:5]) / 5,
             label="first 5 trials")
    plt.plot(range(100), sum(bandit_rewards_list) / 10,
             label="first 10 trials")
    plt.xlabel('s')
    plt.ylabel('avg. reward after s steps')
    plt.title('MultiArmedBandit Average Rewards Comparison')
    plt.legend()
    plt.savefig('experiments/2a.png')

    env2 = gym.make('SlotMachines-v0')
    q_rewards_list = []

    for _ in range(10):
        q = QLearning()
        _, rewards = q.fit(env2, steps=100000)
        q_rewards_list.append(rewards)

    plt.figure()
    plt.plot(range(100), sum(bandit_rewards_list) / 10,
             label="bandit first 10 trials")
    plt.plot(range(100), sum(q_rewards_list) / 10,
             label="q learning first 10 trials")
    plt.xlabel('s')
    plt.ylabel('avg. reward after s steps')
    plt.title('MultiArmedBandit vs. QLearning Average Rewards Comparison')
    plt.legend()
    plt.savefig('experiments/2b.png')


def question_3a():
    env1 = gym.make('FrozenLake-v0')
    q1_rewards_list = []

    for _ in range(10):
        q1 = QLearning(epsilon=0.01)
        _, rewards = q1.fit(env1, steps=100000)
        q1_rewards_list.append(rewards)

    env2 = gym.make('FrozenLake-v0')
    q2_rewards_list = []

    for _ in range(10):
        q2 = QLearning(epsilon=0.5)
        _, rewards = q2.fit(env2, steps=100000)
        q2_rewards_list.append(rewards)

    plt.figure()
    plt.plot(range(100), sum(q1_rewards_list) / 10,
             label="first 10 trials with epsilon = 0.01")
    plt.plot(range(100), sum(q2_rewards_list) / 10,
             label="first 10 trials with epsilon = 0.5")
    plt.xlabel('s')
    plt.ylabel('avg. reward after s steps')
    plt.title('QLearning Average Rewards Comparison with Varying Epsilon')
    plt.legend()
    plt.savefig('experiments/3a.png')


if __name__ == "__main__":

    # questions_2a_and_2b()
    # question_3a()
