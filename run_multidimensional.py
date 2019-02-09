import run_policy
import argparse
import gym

from implementations.algorithms import TD3
from implementations.algorithms import DDPG
from implementations.utils import replay_buffer

import gym_multi_dimensional
from gym_multi_dimensional.visualization import vis_2d

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name",default="Random")
    parser.add_argument("--policy_directory", default="policies")
    parser.add_argument("--dimensions", default=2, type=int)
    parser.add_argument("--max_timesteps", default=50, type=int)
    parser.add_argument("--render", default=True, type=bool)
    parser.add_argument("--verbose", default=True, type=bool)
    parser.add_argument("--continuous", default=True, type=bool)
    parser.add_argument("--acceleration", default=True, type=bool)

    args = parser.parse_args()

    environment = gym_multi_dimensional.dynamic_register(n_dimensions=args.dimensions,
            env_description={},continuous=args.continuous,acceleration=args.acceleration)

    replay_buffer = run_policy.run_policy(policy_name=args.policy_name,
            policy_directory=args.policy_directory,
            environment=environment,
            max_timesteps=args.max_timesteps,
            render=args.render,
            verbose=args.verbose)

    vis_2d.visualize_RB(replay_buffer)

    
    if args.policy_name == "Random":
        pass
    else:
        
        env = gym.make(environment)
        
        state_dim = 1
        for dim_length in env.observation_space.shape:
            state_dim *= dim_length
        action_dim = 1
        for dim_length in env.action_space.shape:
            action_dim *= dim_length
        max_action = float(env.action_space.high[0])

        env.close()

        if args.policy_name == "TD3":
            policy = TD3.TD3(state_dim,action_dim,max_action)
        elif args.policy_name == "DDPG":
            policy = DDPG.DDPG(state_dim,action_dim,max_action)

        policy.load(args.policy_name + "_" + environment,"policies")
        Q_values = policy.Q_values(replay_buffer)
        vis_2d.visualize_Q(Q_values)
        vis_2d.visualize_Q2(Q_values)
