import numpy as np
import torch
import gym
import os
import argparse
import matplotlib.pyplot as plt

from implementations.algorithms import DDPG
from implementations.algorithms import TD3
from implementations.utils import replay_buffer

def visualize_training(evaluations, freq=1, save=False, path=''):
    x = np.arange(0, freq * len(evaluations), freq)

    plt.errorbar(x, evaluations[:, 0], evaluations[:, 1], fmt="--o")
    plt.title("Average reward per step")

    if save:
        plt.savefig(path + "/scores.png")

    plt.show()


# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, env, eval_episodes=200):
    avg_reward = 0.
    total_steps = 0

    records = np.array([])
    steps = np.array([])

    for i in range(eval_episodes):

        episode_reward = 0
        episode_steps = 0

        state = env.reset()
        done = False

        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_steps += 1

            if episode_steps > env._max_episode_steps:
                done = True

        total_steps += episode_steps

        records = np.append(records, episode_reward)
        steps = np.append(steps, episode_steps)

    avg_reward = np.sum(records) / total_steps
    error = np.std(records / steps)

    print ("---------------------------------------")
    print ("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print ("---------------------------------------")

    return np.array([avg_reward, error])


def learn(policy_name="DDPG",
            policy_directory='policies',
            evaluations_directory='evaluations',
            visualizations_directory='visualizations',
            save=False,
            seed=0,
            environment=None,
            eval_freq=5e3,
            exploration_timesteps=1e3,
            learning_timesteps=1e4,
            buffer_size=5000,
            new_exp=True,
            expl_noise=0.1,
            batch_size=64,
            discount=0.99,
            learning_rate=1e-4,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            filter=None):

    q_values = []
    q_pi_values = []
    env = gym.make(environment)

    file_name = "%s_%s" % (policy_name, environment)

    print ("---------------------------------------")
    print ("Settings: %s" % (file_name))
    print ("---------------------------------------")

    # Set seeds
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    if policy_name == "TD3":
        policy = TD3.TD3(state_dim, action_dim, max_action,learning_rate=learning_rate)
    else:
        policy = DDPG.DDPG(state_dim, action_dim, max_action,learning_rate=learning_rate)

    rb = replay_buffer.ReplayBuffer(buffer_size)

    # Evaluate untrained policy
    evaluations = [evaluate_policy(policy,env)]

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_timesteps = 0
    episode_reward = 0
    done = True

    while total_timesteps < exploration_timesteps:

        if done:
            # Reset environment
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

            if filter is not None:
                while(filter.isIn(obs)):
                    obs = env.reset()

        action = env.action_space.sample()

        # Perform action
        new_obs, reward, done, _ = env.step(action)

        if filter is not None:
            if filter.isOut(new_obs):
                rb.push(obs, action, reward, done, new_obs)
                episode_timesteps += 1
                total_timesteps += 1
                timesteps_since_eval += 1
                episode_reward += reward
                obs = new_obs
        else:
            rb.push(obs, action, reward, done, new_obs)
            episode_reward += reward
            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1
            obs = new_obs

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_timesteps = 0
    episode_reward = 0
    done = True

    while total_timesteps < learning_timesteps:

        if done:
            if total_timesteps != 0:
                print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(total_timesteps, episode_num, episode_timesteps, episode_reward))

                if policy_name == "TD3":
                    policy.train(rb, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)
                else:
                    policy.train(rb, episode_timesteps, batch_size, discount, tau)

                # Evaluate episode
                if timesteps_since_eval >= eval_freq:
                    timesteps_since_eval %= eval_freq
                    evaluations.append(evaluate_policy(policy,env))
                    np.save(evaluations_directory + "/%s" % (file_name), evaluations)
                    if state_dim <= 2:
                        q_values.append(policy.get_Q_values(env, 20))
                        q_pi_values.append(policy.get_Q_values(env, 10,pi=True))

                # Reset environment
                obs = env.reset()
                done = False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

        # Select action randomly or according to policy
        action = policy.select_action(np.array(obs))

        if expl_noise != 0:
            action = (action + np.random.normal(0, expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        episode_reward += reward

        # Push experience to rb if in exploration phase and in exploitation if new_exp is True
        if new_exp == True:
            rb.push(obs, action, reward, done, new_obs)

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1

    # Final evaluation
    evaluations.append(evaluate_policy(policy,env))
    evaluations = np.array(evaluations)

    if not os.path.exists(policy_directory):
        os.makedirs(policy_directory)

    policy.save("%s" % (file_name), directory=policy_directory)

    if not os.path.exists(evaluations_directory):
        os.makedirs(evaluations_directory)

    np.save(evaluations_directory + "/%s" % (file_name), evaluations)

    visualize_training(evaluations, eval_freq, save, visualizations_directory)

    return rb, q_values, q_pi_values

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name",default="DDPG")
    parser.add_argument("--policy_directory", default="policies")
    parser.add_argument("--seed", default=0, type=int)              #seed
    parser.add_argument("--environment", default="MountainCarContinuous-v0")
    parser.add_argument("--eval_freq", default=5e3, type=float)     #how often (time steps) we evaluate
    parser.add_argument("--exploration_timesteps", default=1e3, type=int) #random steps at the beginning
    parser.add_argument("--learning_timesteps", default=1e4, type=int)
    parser.add_argument("--buffer_size", default=5000, type=int)
    parser.add_argument("--no-new-exp", dest='new_exp', action="store_false")
    parser.set_defaults(new_exp=True)
    parser.add_argument("--expl_noise", default=0.1, type=float)    #noise
    parser.add_argument("--batch_size", default=64, type=int)      #learning batch
    parser.add_argument("--discount", default=0.99, type=float)     #discount factor
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--tau", default=0.005, type=float)         #target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  #noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)    #range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       #frequency of delayed policy updates
    
    args = parser.parse_args()

    learn(policy_name=args.policy_name,
            policy_directory=args.policy_directory,
            seed=args.seed,
            environment=args.environment,
            eval_freq=args.eval_freq,
            exploration_timesteps=args.exploration_timesteps,
            learning_timesteps=args.learning_timesteps,
            buffer_size=args.buffer_size,
            new_exp=args.new_exp,
            expl_noise=args.expl_noise,
            batch_size=args.batch_size,
            discount=args.discount,
            learning_rate=args.learning_rate,
            tau=args.tau,
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip,
            policy_freq=args.policy_freq)

