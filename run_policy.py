import gym
import numpy as np

from implementations.algorithms import TD3
from implementations.algorithms import DDPG
from implementations.utils import replay_buffer


def run_policy(policy_name="Random",policy_directory=None,environment=None,
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

    rb = replay_buffer.ReplayBuffer(5000)
    old_state = None
    avg_reward = 0.
    for _ in range(max_timesteps):
        old_state = env.reset()
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
            rb.push(old_state, action, reward, done, state)
            old_state = state
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
    return rb
