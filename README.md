# Kinova Jaco gen3 RL Environment for 2 finger
This repository contains a complete environment for Reinforcement Learning that can be run both in simulation and on a physical Kinova arm. We provide the base robot environment under ```jaco-gym/jaco_gym/envs/robot_env.py``` and also provide an example task environment under ```jaco-gym/jaco_gym/envs/task_envs/stack_cups_gazebo.py``` that facilitates training for a cup-stacking robot (shown below). Additonally, this repo contains implementations of numerous RL training methods that can be used for your learning task under the ```rlkit``` submodule.


![Jaco Gazebo](/jaco_world.png)

At a Glance 
|---------------|
| [Repo Layout](#repo-layout) - describing each folder in our repository and the basic setup | 
| [Enviornment Details](#environment-details) - explaining our gym environemnt setup, action and state spaces |
| [Installation Instructions](https://github.com/pranavraj575/jacorl/blob/master/setup/README.md) - instructions for how to set up and run the code |
| [Agent Training](#train-the-agent) - instructions for training the robot |

## Repo Layout
The repository contains the following sub-folders:
* <b>jaco-gym</b> - contains robot enviornment, task environment, and sample training scripts
* <b>ros-kortex</b> - contains launch files and world file (jaco.world is our world file for our cup training example that contains a table and 3 cups that spawn randomly upon reset)
* <b>rlkit</b> - contains packages for training robot (SAC, Q-learning, etc.), cloned from https://github.com/rail-berkeley/rlkit
* <b>ros-numpy</b> - contains packages necessary for saving camera images to numpy (camera attached to robot joint), cloned from https://github.com/eric-wieser/ros_numpy

## Environment Details

<b>Action space</b> 

The action space is a 7d array of values between -1 and 1, where indices 0-5 represent movement displacements of each robot joint, and index 6 represents the new position of the fingers (1 = fingers closed, -1 = fingers opened). To prevent large jumps in robot position, we define a maximum angular movement each of the 6 non-finger joints can travel in a given step (self.differences), and a 1 in a given joint position represents moving that joint the maximum angular step-distance, while a -1 in that joint posion means rotating the joint backwards the maxmimum amount for a given step. 

- ex. [0,0,0,0,0,0,1] = keeping the robot in the same position, but pinching the fingers
- ex. [1,1,1,1,1,1,-1] = stepping each joint forward by the maximum angular movement (which is really just a relatively small movement in each joint to prevent large jumps), and unpinching the fingers

<b>Useful Features</b>

Unlike other Kinova implementations that are specific to either a physical robot or a simulation, we provide an implementation flexible enough to work in both environments. The robot enviornment also provides many useful functions that we have implemented without any ROS built-ins (just with pure trig and other computations), so they can be called in both the sim and real world, including:

- Built-in <b>camera</b> mounted on the robot joint that stores Image objects through ```self.camera_data```, and functions to <b>save the current camera image</b> through the ```save_image``` function that is supported by the included ros_numpy sub-directory
- Functions to get the angular position of each of the joints, and also convert each of these to cartesian coordinates to determine the <b>exact (x,y,z) coordinates of all of the robot joints.</b>
- Functions to <b>detect the robot colliding</b> with its joints 

<br /> 

![Jaco Gazebo](/env_features.png)

Camera capture from the simulated robot (left), and bounding box of joint on the simulated robot (right)

## Installation

See [installation details](https://github.com/pranavraj575/jacorl/blob/master/setup/README.md) here

## Train the agent

In terminal 1, cd into the ros-kortex directory and launch the simulated robot environment:
```bash
roslaunch kinova_gazebo robot_launch_noRender_noSphere.launch kinova_robotType:=j2n6s300 
```

In terminal 2, run the python robot training script:
```bash
python3 scripts/1_train_ppo2.py
```

## Enjoy a trained agent

In terminal 1:
```bash
roslaunch kinova_gazebo robot_launch.launch kinova_robotType:=j2n6s300
```

Uncomment this line in jaco_gym/envs/jaco_gazebo_action_env.py
```python
self.robot.move_sphere(self.target_vect)
```

In terminal 2:
```bash
python3 scripts/2_enjoy_ppo2.py
```

## Plot learning curves

```bash
python3 scripts/3_plot_results.py
```


## Train with Stable Baselines


In terminal 1:
```bash
roslaunch kinova_gazebo robot_launch_noRender_noSphere.launch kinova_robotType:=j2n6s300 
```

In terminal 2:
```bash
cd stable-baselines-zoo/
python3 train.py --algo ppo2 --env JacoGazebo-v1 -n 100000 --seed 0 --log-folder logs/ppo2/JacoGazebo-v1_100000/ &> submission_log/log_ppo_jaco.run
python3 train.py --algo sac --env JacoGazebo-v1 -n 100000 --seed 0 --log-folder logs/sac/JacoGazebo-v1_100000/
python3 train.py --algo td3 --env JacoGazebo-v1 -n 100000 --seed 0 --log-folder logs/td3/JacoGazebo-v1_100000/
```



## Enjoy a trained agent with Stable Baselines

In terminal 1:
```bash
roslaunch kinova_gazebo robot_launch.launch kinova_robotType:=j2n6s300
```

Uncomment this line in jaco_gym/envs/jaco_gazebo_action_env.py
```python
self.robot.move_sphere(self.target_vect)
```

In terminal 2:
```bash
cd stable-baselines-zoo/
python3 enjoy.py --algo ppo2 --env JacoGazebo-v1 -f logs/ --exp-id 0 -n 2000
```

## Plot stable_baselines results

```bash
python3 plot_results.py -f logs/ppo2/JacoGazebo-v1_1/
```

## Supported systems
Tested on:
- Ubuntu 18.04; ROS Melodic; Python 3.6.9
- Ubuntu 20.04; ROS Noetic; Python 3.8.10
