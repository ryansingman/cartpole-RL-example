import gym
import numpy as np

from neural_net.policy import Policy
from neural_net.trainer import Trainer

from tqdm import tqdm
import pdb

NUM_EPISODES = 10000
NUM_STEPS = 475

DISPLAY_MODE = False 

ENV_NAME = 'CartPole-v1'
REWARD_THRESH = {'CartPole-v1': 450}

if __name__ == '__main__':

    # create environment
    env = gym.make(ENV_NAME)

    # initialize policy and trainer
    policy_config = {'num_hidden_layers': 1, 'hidden_layer_sizes': (128,), 'dropout': 0.6, \
                     'max_epsilon': 0.9, 'min_epsilon': 0.02, 'eps_rate': 0.95, 'max_episodes': NUM_EPISODES}
    optim_config = {'initial_lr': 1e-2, 'min_lr': 1e-5, 'lr_rate': 0.85,  'gamma': 0.99}

    policy = Policy(env, policy_config)
    trainer = Trainer(policy, optim_config)

    # iterate through episodes
    with tqdm(total = NUM_EPISODES) as pbar:
        for episode in range(NUM_EPISODES):
            avg_reward = 0
            observation = env.reset()
            for t in range(NUM_STEPS):
                if DISPLAY_MODE and episode % 100 == 0:
                    env.render()

                # take action
                action = policy.action(observation)
                observation, reward, done, _ = env.step(action.item())

                # update reward
                policy.update_reward(reward) 

                if done:
                    break

            loss = trainer.update_policy()

            if episode % 100 == 0 and episode >= 100:
                avg_reward = np.mean(policy.reward_history[episode-50:episode])
                print("Average rewards at episode {}: {}".format(episode, avg_reward))

            if avg_reward > REWARD_THRESH[ENV_NAME]:
                break

            pbar.update(1)

    # run episode w/ render
    policy.start_testing_mode()

    done = False
    observation = env.reset()
    alive_steps = 0
    while not done:
        # render environment
        env.render()

        # take action
        action = policy.action(observation)
        observation, _, done, _ = env.step(action.item())

        alive_steps += 1

    print("Survived {} steps".format(alive_steps))

    env.close()
