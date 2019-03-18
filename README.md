# Being Prepared For Anything
Algorithms, experiments, and hyperparameters accompanying "Being Prepared For Anything: Evaluating task-agnostic exploration for fixed-batch learning of arbitrary future tasks"

## Requirements
We use:
- [Pytorch](https://github.com/pytorch/pytorch) v1.0.1
- [OpenAI Gym](https://github.com/openai/gym) v0.9.1
- [Mujoco](http://www.mujoco.org/) 130 and [Mujoco-Py](https://github.com/openai/mujoco-py) v0.5.7



## Framework
Individual experiments are represented by their parameter files, located in 'params/domain_name'. These files specify all parameters for the experiment, including the environment to be instantiated. The base parameter file specifies the general parameters for each domain, while the more specific files include the parameters for individual experiments. 'agent.py' implements all agent classes, while session implements the session classes, which run episodes, allowing the agents to interact with the environment. Task specifies user defined rewards, which include intrinsic rewards.

## Environments
The simulated Mujoco environments evaluated on are:
- 'HalfCheetah-v1'
- 'Walker2d-v1'
- 'Hopper-v1'

We also evaluate on a real robot, the [Franka Emika Panda](https://www.franka.de/panda/).


## Running Experiments

To run an experiment, run the ‘run’ script and specify a parameter file. For example, to run RND as an exploration method on half cheetah:

```./run params/cheetah/explore_rnd.py```


