import gym
import numpy as np
import argparse

import gym_multi_dimensional

from implementations.algorithms import TD3
from implementations.algorithms import DDPG


def run_policy(policy_name="Random",policy_directory="policies",environment="",
        max_timesteps=50,render=True,verbose=True):

    env = gym.make(environment)

    state_dim = 1
    for dim_length in env.observation_space.shape:
        state_dim *= dim_length
    action_dim = 1
    for dim_length in env.action_space.shape:
        action_dim *= dim_length
    max_action = float(env.action_space.high[0])

    if policy_name == "TD3":
        policy = TD3.TD3(state_dim,action_dim,max_action)
        policy.load("TD3_MultiDimensional-v2_0","policies")
    elif policy_name == "DDPG":
        policy = TD3.TD3(state_dim,action_dim,max_action)
        policy.load("DDPG_MultiDimensional-v2_0","policies")
    elif policy_name == "Random":
        pass


    avg_reward = 0.
    for _ in range(max_timesteps):
        state = env.reset()
        done = False
        i=0
        cur_reward = 0
        while not done:
            if render==True:
                env.render()
            if policy_name == "Random":
                action = env.action_space.sample()
            else:
                action = policy.select_action(np.array(state))
            state, reward, done, info = env.step(action)
            cur_reward += reward

            if verbose==True:
                print(state,reward,done,info)
        
            avg_reward += reward
            i+=1

        if verbose==True:
            print("Episode finished after {} timesteps, final reward : {}".format(i+1,cur_reward))


    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (max_timesteps, avg_reward))
    print("---------------------------------------")

    env.close()


if __name__ == "__main__":
    
    id = gym_multi_dimensional.dynamic_register(n_dimensions=2,
            env_description={},continuous=True,acceleration=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name",default="Random")
    parser.add_argument("--policy_directory", default="policies")
    parser.add_argument("--environment", default=id)
    parser.add_argument("--max_timesteps", default=50)
    parser.add_argument("--render", default=True)
    parser.add_argument("--verbose", default=True)

    args = parser.parse_args()

    run_policy(policy_name=args.policy_name,
            policy_directory=args.policy_directory,
            environment=args.environment,
            max_timesteps=args.max_timesteps,
            render=args.render,
            verbose=args.verbose)
