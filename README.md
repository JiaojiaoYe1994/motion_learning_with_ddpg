# Motion Planning with Reinforcement Learning


## Humanoid robot Poppy arm reaching movement learning with collision avoidance
This project focuses on training PoppyHumanoid to reach a target on the table. The target could appear at every possible point on the table. All details about this part of the project can be found in the corresponding report.

The implementation of this projects mainly contains two parts: environment and algorithm. Environment, which can be modeled as Markov Decision Process, is implemented from scratch without using additional reinforcement learning libraries.  

## Dependencies
* Python 2.7
* tensorflow 1.6
* numpy 1.11.0
* matplotlib 1.5.1
* tensorflow 1.4.0
* pypot 2.11.0rc5
* vrep simulator

## Installation
pip install tensorflow-gpu==1.4.0
pip install pypot

## Usage

 git clone https://github.com/JiaojiaoYe1994/robot_motion_learning_with_reinforcement_learning.git
 
 cd ./robot_motion_learning_with_reinforcement_learning/

### 1. Training
python train.py, and try python train.py -h for possible input arguments.
For example, 

### 2. Testing
python eval.py, and try python eval.py -h for possible input arguments.
For example


## License Information
MIT



