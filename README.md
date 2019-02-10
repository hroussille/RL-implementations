# RL-implementations

The code from this repository is mostly comming from : [https://github.com/whikwon/TD3](https://github.com/whikwon/TD3)

After an extensive cleanup we are mostly using the following files with some modifications :
* main.py (learn_policy.py)
* TD3.py  (TD3.py)
* DDPG.py (DDPG.py)

Those modified implementations are mostly used to test our [custom gym environment](https://github.com/hroussille/RL-evaluation-environment)

And to provide additional functionalities related with :

* Benchmarking
* Constraint state space
* Custom Replay Buffer with visualization
* and more..

## Usage Examples

### on our custom gym environment

To quickly learn a policy with DDPG on our 2D env :
```sh
python learn_multidimensional.py --max_timesteps 10000 --policy_name DDPG
```
To run a policy that has been learn, and see the exploration and the action values:
```sh
python run_multidimensional.py --policy_name DDPG --max_episodes 100
```
you can add to the previous command line to run it quickly and only see the generated graphs :
```sh
--no-render --quiet
```
To analyse with an existing policy on a random generated replay buffer :
```sh
python analyze_multi_policy.py --max_episodes 1000 --buffer_size=500000 --batch_size 1000 --quiet --no-render --policy_name DDPG
```

### on standards gym environments

```sh
python learn_policy.py --policy_name DDPG --environment 'MountainCarContinuous-v0'
python run_policy.py --policy_name DDPG --environment 'MountainCarContinuous-v0'
```
