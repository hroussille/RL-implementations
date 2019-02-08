import gym
import gym_multi_dimensional
import numpy as np
from implementations.algorithms import TD3


id = gym_multi_dimensional.dynamic_register(n_dimensions=2,env_description={},continuous=True,acceleration=True)
env = gym.make(id)

state_dim = env.observation_space.shape[0]*env.observation_space.shape[1]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

policy = TD3.TD3(state_dim,action_dim,max_action)
policy.load("TD3_MultiDimensional-v2_0","pytorch_models")

avg_reward = 0.
for _ in range(50):
    state = env.reset()
    done = False
    i=0
    while not done and i< env._max_episode_steps:
        #env.render()
        action = policy.select_action(np.array(state))
        state, reward, done, _ = env.step(action)
        avg_reward += reward
        i+=1

print("---------------------------------------")
print("Evaluation over %d episodes: %f" % (50, avg_reward))
print("---------------------------------------")

env.close()
