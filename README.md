# RL-implementations

The DDPG / TD3 implementations are originally from : [this repository](https://github.com/whikwon/TD3)

## Usage Examples

### Learn a policy

_**Warning:**  This code is intended to be run from a [higher level implementations]([https://github.com/schott97l/RL_analysis](https://github.com/schott97l/RL_analysis))._

To learn a policy with one of the algorithm , one must run :

```sh
python learn_policy.py --policy_name DDPG --environment 'MountainCarContinuous-v0'
```
For a comprehensive summary of all the parameters and their description please run :
```sh
python learn_policy.py --help
```

### Run a policy

_**Warning:**  This code is intended to be run from a [higher level implementations]([https://github.com/schott97l/RL_analysis](https://github.com/schott97l/RL_analysis))._

```sh
python run_policy.py --policy_name DDPG --environment 'MountainCarContinuous-v0'
```

### Reminder
This reposity is still under active development and some functionalities might break.
