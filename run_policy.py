import gym
import numpy as np
import argparse

from implementations.algorithms import TD3
from implementations.algorithms import DDPG
from implementations.utils import replay_buffer


def run_policy(algorithm="Random",
        policy_directory="policies",
        environment=None,
        max_episodes=50,
        max_timesteps=10000,
        buffer_size=5000,
        render=True,
        verbose=True):

    env = gym.make(environment)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    if algorithm == "Random":
        pass
    else:
        if algorithm == "TD3":
            policy = TD3.TD3(state_dim,action_dim,max_action)
        elif algorithm == "DDPG":
            policy = DDPG.DDPG(state_dim,action_dim,max_action)
        policy.load(algorithm + "_" + environment,policy_directory)

    rb = replay_buffer.ReplayBuffer(buffer_size)
    old_state = None
    avg_reward = 0.
    timestep=0
    for _ in range(max_episodes):
        old_state = env.reset()
        done = False
        if timestep>max_timesteps:
            done = True
        i=0
        cur_reward = 0
        while not done:
            if render:
                env.render()
            if algorithm == "Random" :
                action = env.action_space.sample()
            else:
                action = policy.select_action(np.array(old_state))
            state, reward, done, info = env.step(action)
            rb.push(old_state, action, reward, done, state)
            old_state = state
            cur_reward += reward

            if verbose:
                print(state,reward,done,info)
        
            avg_reward += reward
            i+=1
            timestep +=1
            if timestep>max_timesteps:
                break;

        if verbose:
            print("Episode finished after {} timesteps, final reward : {}".format(i+1,cur_reward))

    if verbose:
        print("---------------------------------------")
        print("Evaluation over %d episodes: %f" % (max_episodes, avg_reward))
        print("---------------------------------------")

    env.close()
    return rb

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm",default="Random")
    parser.add_argument("--policy_directory", default="results/models")
    parser.add_argument("--environment", default="MountainCarContinuous-v0")
    parser.add_argument("--max_episodes", default=50, type=int)
    parser.add_argument("--max_timesteps", default=1e4, type=int)
    parser.add_argument("--buffer_size", default=5000, type=int)
    parser.add_argument('--quiet', dest='verbose', action='store_false')
    parser.add_argument('--no_render', dest='render', action='store_false')

    parser.set_defaults(verbose=True)
    parser.set_defaults(render=True)

    args = parser.parse_args()
    
    run_policy(algorithm=args.algorithm,
            policy_directory=args.policy_directory,
            environment=args.environment,
            max_episodes=args.max_episodes,
            max_timesteps=args.max_timesteps,
            buffer_size=args.buffer_size,
            render=args.render,
            verbose=args.verbose)
