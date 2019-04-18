import numpy as np
import torch
import gym
import os
import argparse
import matplotlib.pyplot as plt

import json

from implementations.algorithms import DDPG
from implementations.algorithms import TD3
from implementations.utils import replay_buffer

def populate_output_dir(path, exist):
    if exist is False:
        os.makedirs(path)

    os.makedirs(path + "/models")
    os.makedirs(path + "/visualizations")
    os.makedirs(path + "/logs")


def setup_output_dir(path):

    exist = os.path.exists(path)

    if exist:
        
        if os.path.isdir(path) is False:
            print("Output path : {} already exist and is not a directory".format(path))
            return False

        if len(os.listdir(path)) != 0:
            print("Output directory : {} already exists and is not empty".format(path))
            return False

    populate_output_dir(path, exist)
    return True

def save_arguments(args, path):
    with open(path + '/arguments.txt', 'w') as file:
        file.write(json.dumps(args))

def visualize_training(evaluations, freq=1, save=False, path=''):
    x = np.arange(0, freq * len(evaluations), freq)

    plt.plot(x, evaluations[:, 0], "-o")
    plt.title("Average reward per step",fontsize=12)
    plt.tick_params(labelsize=12)

    if save:
        plt.savefig(path + "/scores.png")
    else:
        plt.show()


# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, env, verbose,eval_episodes=100):
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

    if verbose:
        print ("---------------------------------------")
        print ("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
        print ("---------------------------------------")

    return np.array([avg_reward, error])


def learn(algorithm="DDPG",
            output='results',
            save=False,
            seed=0,
            environment=None,
            eval_freq=5e3,
            exploration_timesteps=1e3,
            exploration_mode="sequential",
            learning_timesteps=1e4,
            buffer_size=5000,
            new_exp=True,
            expl_noise=0.1,
            batch_size=64,
            discount=0.99,
            actor_dim=(40,30),
            critic_dim=(40,30),
            learning_rate=1e-4,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            filter=None,
            verbose=True,
            render=True):

    q_values = []
    pi_values = []
    env = gym.make(environment)
    eval_env = gym.make(environment)

    file_name = "%s_%s" % (algorithm, environment)

    if verbose:
        print ("---------------------------------------")
        print ("Settings: %s" % (file_name))
        print ("---------------------------------------")

    # Set seeds
    env.seed(seed)
    eval_env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    if algorithm == "TD3":
        policy = TD3.TD3(state_dim, action_dim, max_action, actor_dim=actor_dim,
                critic_dim=critic_dim, learning_rate=learning_rate)
    else:
        policy = DDPG.DDPG(state_dim, action_dim, max_action,actor_dim=actor_dim,
                critic_dim=critic_dim, learning_rate=learning_rate)

    rb = replay_buffer.ReplayBuffer(buffer_size)

    # Evaluate untrained policy
    evaluations = [evaluate_policy(policy,eval_env,verbose)]
    if state_dim <= 2:
        q_values.append(policy.get_Q_values(env, 20))
        pi_values.append(policy.get_Pi_values(env, 10))


    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_timesteps = 0
    episode_reward = 0
    done = True

    while total_timesteps < exploration_timesteps:

        push = True

        if exploration_mode == "uniform":
            obs = env.observation_space.sample()

        if done:
            # Reset environment
            if exploration_mode == "sequential":
                obs = env.reset()

            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

            if filter is not None:
                while(filter.isIn(obs)):
                    if exploration_mode == "sequential":
                        obs = env.reset()
                    elif exploration_mode == "uniform":
                        obs = env.observation_space.sample()

        if render:
            env.render()

        action = env.action_space.sample()
        
        # Perform action
        if exploration_mode == "sequential":
            new_obs, reward, done, _ = env.step(action)

        elif exploration_mode == "uniform":
            new_obs, reward, done, _ = env.sample_step(obs, action)

        if filter is not None:
            if filter.isIn(new_obs) or filter.isIn(obs):
                push = False

        if push is True:
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
                if verbose:
                    print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(total_timesteps, episode_num, episode_timesteps, episode_reward))

                if algorithm == "TD3":
                    policy.train(rb, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)
                else:
                    policy.train(rb, episode_timesteps, batch_size, discount, tau)
                
                # Reset environment
                obs = env.reset()
                done = False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

        # Evaluate episode
        if timesteps_since_eval >= eval_freq:
            timesteps_since_eval %= eval_freq
            evaluations.append(evaluate_policy(policy,eval_env,verbose))
            if state_dim <= 2:
                q_values.append(policy.get_Q_values(env, 20))
                pi_values.append(policy.get_Pi_values(env, 10))
        
        if render:
            env.render()
       
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
    evaluations.append(evaluate_policy(policy,eval_env,verbose))
    evaluations = np.array(evaluations)
    if state_dim <= 2:
        q_values.append(policy.get_Q_values(env, 20))
        pi_values.append(policy.get_Pi_values(env, 10))

    if save:
        if not os.path.exists(output+"/models"):
            os.makedirs(output+"/models")

        policy.save("%s" % (file_name), directory=output+"/models/")

        if not os.path.exists(output + "/logs"):
            os.makedirs(output + "/logs")

        np.save(output + "/logs/evaluations", evaluations)
        np.save(output + "/logs/q_values", q_values)
        np.save(output + "/logs/pi_values", pi_values)
        np.save(output + "/logs/replay_buffer", rb)

    if save:
        if not os.path.exists(output + "/visualizations"):
            os.makedirs(output + "/visualizations")

    visualize_training(evaluations, eval_freq, save, output+"/visualizations")

    env.close()
    eval_env.close()

    return rb, q_values, pi_values

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm",default="DDPG")
    parser.add_argument("--output", default="results")
    parser.add_argument('--save', dest='save', action='store_true')
    parser.add_argument("--seed", default=0, type=int)              #seed
    parser.add_argument("--environment", default="MountainCarContinuous-v0")
    parser.add_argument("--eval_freq", default=5e3, type=float)     #how often (time steps) we evaluate
    parser.add_argument("--exploration_timesteps", default=1e3, type=int) #random steps at the beginning
    parser.add_argument("--exploration_mode", default="sequential", type=str)
    parser.add_argument("--learning_timesteps", default=1e4, type=int)
    parser.add_argument("--buffer_size", default=5000, type=int)
    parser.add_argument("--no_new_exp", dest='new_exp', action="store_false")
    parser.add_argument("--expl_noise", default=0.1, type=float)    #noise
    parser.add_argument("--batch_size", default=64, type=int)      #learning batch
    parser.add_argument("--discount", default=0.99, type=float)     #discount factor
    parser.add_argument("--actor_hl1", default=40, type=int)    #actor hidden layer 1
    parser.add_argument("--actor_hl2", default=30, type=int)    #actor hidden layer 2
    parser.add_argument("--critic_hl1", default=40, type=int)   #critic hidden layer 1
    parser.add_argument("--critic_hl2", default=30, type=int)   #critic hidden layer 2
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--tau", default=0.005, type=float)         #target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  #noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)    #range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       #frequency of delayed policy updates
    parser.add_argument('--quiet', dest='verbose', action='store_false')
    parser.add_argument("--no_render", dest="render", action="store_false")       #rednering

    parser.set_defaults(save=False)
    parser.set_defaults(verbose=True)
    parser.set_defaults(new_exp=True)
    parser.set_defaults(render=True)
    
    args = parser.parse_args()

    if args.save:
        if not setup_output_dir(args.output):
            exit()

    learn(algorithm=args.algorithm,
            output=args.output,
            save=args.save,
            seed=args.seed,
            environment=args.environment,
            eval_freq=args.eval_freq,
            exploration_timesteps=args.exploration_timesteps,
            exploration_mode=args.exploration_mode,
            learning_timesteps=args.learning_timesteps,
            buffer_size=args.buffer_size,
            new_exp=args.new_exp,
            expl_noise=args.expl_noise,
            batch_size=args.batch_size,
            discount=args.discount,
            actor_dim=(args.actor_hl1,args.actor_hl2),
            critic_dim=(args.critic_hl1,args.critic_hl2),
            learning_rate=args.learning_rate,
            tau=args.tau,
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip,
            policy_freq=args.policy_freq,
            verbose=args.verbose,
            render=args.render)

    if args.save:
        save_arguments(vars(args), args.output)

