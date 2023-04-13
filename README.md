## Project Brief
Using DQN for the gym lunar lander environment. 
Everything works but I made a few iffy design choices out of laziness.

## Installation
pip install .

You may also need to run **apt install python-opengl**.

## Example Run Arguments

To be run at the root of the project.

- python3 src/lunar_lander_dqn/run.py --tune config/tune.yaml --num_episodes 1000
- python3 src/lunar_lander_dqn/run.py --train config/train.yaml --num_episodes 1000
- python3 src/lunar_lander_dqn/run.py --run model/model.pth --num_episodes 10
