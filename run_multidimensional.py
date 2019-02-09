import run_policy
import argparse

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
