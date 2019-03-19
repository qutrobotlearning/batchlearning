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

Upon completion, the '.data' folder will be created and the data stored within. You can then train an agent on this offline data by simply pointing the run file to the corresponding parameter file. For example, to train a TD3 agent on the offline data produced by the previous command:

```./run params/cheetah/offline_td3_on_rnd.py```

Note that tensorboard log files will be stored in the 'runs' directory, allowing you to view the progress of training.



