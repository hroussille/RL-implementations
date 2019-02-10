import numpy as np
import torch
import gym
import os

from implementations.algorithms import DDPG
from implementations.algorithms import TD3
from implementations.utils import replay_buffer

# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, env, eval_episodes=10):
    avg_reward = 0.
    steps = 0

    for i in range(eval_episodes):

        print("Evaluation : {}".format(i))

        state = env.reset()
        done = False

        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = env.step(action)
            avg_reward += reward
            steps = steps + 1

            if steps > env._max_episode_steps:
                done = True

    avg_reward /= eval_episodes

    print ("---------------------------------------")
    print ("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print ("---------------------------------------")
    return avg_reward


def learn_policy(policy_name="DDPG",
            policy_directory="policies",
            seed=0,
            environment=None,
            eval_freq=5e3,
            start_timesteps=1e3,
            max_timesteps=1e4,
            buffer_size=5000,
            new_exp=True,
            expl_noise=0.1,
            batch_size=100,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2):

    env = gym.make(environment)

    file_name = "%s_%s" % (policy_name, id)
    print ("---------------------------------------")
    print ("Settings: %s" % (file_name))
    print ("---------------------------------------")

    if not os.path.exists("./results"):
            os.makedirs("./results")

    # Set seeds
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    state_dim = 1
    for dim_length in env.observation_space.shape:
        state_dim *= dim_length
    action_dim = 1
    for dim_length in env.action_space.shape:
        action_dim *= dim_length
    max_action = float(env.action_space.high[0])

    # Initialize policy
    if policy_name == "TD3":
        policy = TD3.TD3(state_dim, action_dim, max_action)
    else:
        policy = DDPG.DDPG(state_dim, action_dim, max_action)

    rb = replay_buffer.ReplayBuffer(buffer_size)

    # Evaluate untrained policy
    evaluations = [evaluate_policy(policy,env)]

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_timesteps = 0
    episode_reward = 0
    done = True

    Q_values = []

    while total_timesteps < max_timesteps:

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
                        np.save("./results/%s" % (file_name), evaluations)

                # Reset environment
                obs = env.reset()
                done = False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

        # Select action randomly or according to policy
        if total_timesteps < start_timesteps:
                action = env.action_space.sample()
        else:
                action = policy.select_action(np.array(obs))
                if expl_noise != 0:
                        action = (action + np.random.normal(0, expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        #env.render()
        done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
        episode_reward += reward

        # Push experience to rb if in exploration phase and in exploitation if new_exp is True
        if total_timesteps < start_timesteps or (total_timesteps >= start_timesteps and new_exp == True):
            rb.push(obs, action, reward, done_bool, new_obs)

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1

    # Final evaluation
    evaluations.append(evaluate_policy(policy,env))
    if not os.path.exists(policy_directory):
        os.makedirs(policy_directory)
    policy.save("%s" % (file_name), directory=policy_directory)
    if not os.path.exists("results"):
        os.makedirs("results")
    np.save("results/%s" % (file_name), evaluations)
    
    return rb
