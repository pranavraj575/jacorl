# Kinova Jaco2 RL Environment for 2 finger
This repository contains a complete Jaco2 environment for Reinforcement Learning that can be run both in simulation and on a physical Kinova arm. We provide the base robot environment under ```jaco-gym/jaco_gym/envs/robot_env.py``` and also provide an example task environment under ```jaco-gym/jaco_gym/envs/task_envs/stack_cups_gazebo.py``` that facilitates training for a cup-stacking robot (shown below). Additonally, this repo contains implementations of numerous RL training methods that can be used for your learning task under the ```rlkit``` submodule.


![Jaco Gazebo](/jaco_world.png)

## Repo Layout
The repository contains the following sub-folders:
* <b>jaco-gym</b> - contains robot enviornment, task environment, and sample training scripts
* <b>ros-kortex</b> - contains launch files and world file (jaco.world is our world file for our cup training example that contains a table and 3 cups that spawn randomly upon reset)
* <b>rlkit</b> - contains packages for training robot (SAC, Q-learning, etc.), cloned from https://github.com/rail-berkeley/rlkit
* <b>ros-numpy</b> - contains packages necessary for saving camera images to numpy (camera attached to robot joint), cloned from https://github.com/eric-wieser/ros_numpy

## Environment Details

<b>Action space:</b> 7d array of values -1 to 1 indicating how much to move each joint:

For joints 0-5 (all non-finger joints)
- 1 = move the current joint by the max amount in a given "step"
- 0 = hold current position
- -1 = move the current joint in the opposite dirrectino by the max amount

For joint 6 (the finger joint)

- 1 = close both fingers
- 0 = half-closed fingers
- -1 = open both fingers

| Num           | Observation                        | Min   | Max  |
| ------------- | ---------------------------------- | ----- | ---- |
| 0             | joint_1 position                   | -inf  | inf  |
| ...           | ...                                | -inf  | inf  |
| 5             | joint_6 position.                  | -inf  | inf  |
| 6             | joint_finger_1 position            |   0   |  0.8 |
| 7             | joint_1 velocity (rad/s)           | -inf  | inf  |
| 8             | joint_finger_3 angle (rad)         | -inf  | inf  |

(N.m)

## Installation

1. First install [ROS](http://wiki.ros.org/ROS/Installation) if it is not already installed on your machine.

* ROS Melodic on Ubuntu 18.04
* ROS Kinetic on Ubuntu 16.04

Make sure the following line is in the .bashrc file (with `<distro>` replaced by ROS distribution, e.g. `/melodic/`)

```bash
source /opt/ros/<distro>/setup.bash
```

Then complete the following sudo installs:

```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
sudo apt-get install python3-pip
sudo pip3 install rospkg catkin_pkg
sudo apt-get install swig ffmpeg
```

2. Install the robot controllers (maybe remove the stars?)


```bash
sudo apt-get install ros-<distro>-gazebo-ros-control
sudo apt-get install ros-<distro>-ros-controllers*
sudo apt-get install ros-<distro>-trac-ik-kinematics-plugin
sudo apt-get install ros-<distro>-effort-controllers 
sudo apt-get install ros-<distro>-joint-state-controller 
sudo apt-get install ros-<distro>-joint-trajectory-controller 
sudo apt-get install ros-<distro>-controller-*
```
(replace `<distro>` by your ROS distribution, for example `kinetic` or `melodic`)

3. Install and configure your Catkin workspace by following this [link](http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment).

4. Install [jacorl](https://github.com/pranavraj575/jacorl) (this repository) as src. Since this repo contains sub-repositories, you will need to cd into some of the subdirectories and run pip3 install -e . to properly install the required packages (see / copy the lines below):

```bash
cd ~/catkin_ws
rm -rf src
git clone https://github.com/pranavraj575/jacorl src
cd src/jaco-gym
pip3 install -e .
cd ../rlkit
pip3 install -e .
cd ../ros_numpy
pip3 install -e .
cd ../ros_kortex
pip3 install -e .
```

5. Install dependencies for the ros_kortex and ros_kortex_vision package, as indicated [here](https://github.com/Kinovarobotics/ros_kortex) and [here](https://github.com/pranavraj575/jacorl/tree/master/ros_kortex_vision).
```bash
    sudo apt install gstreamer1.0-tools gstreamer1.0-libav libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev gstreamer1.0-plugins-good gstreamer1.0-plugins-base
    sudo apt-get install ros-<distro>-rgbd-launch

``` 

```bash
    sudo apt install python3 python3-pip
    sudo python3 -m pip install conan==1.59
    conan config set general.revisions_enabled=1
    conan profile new default --detect > /dev/null
    conan profile update settings.compiler.libcxx=libstdc++11 default
    cd ~/catkin_ws
    rosdep install --from-paths src --ignore-src -y --rosdistro <distro>
```

(replace `<branch-name>` and `<distro>` with your corresponding ROS distribution, for example `kinetic` or `melodic`)


6. Install the ROS packages and build.

```bash
cd ~/catkin_ws
catkin clean
catkin init
catkin_make
```
Note, the kinova-ros package was adapted from the [official package](https://github.com/Kinovarobotics/kinova-ros).

7. Add the following lines to your bashrc file, or run them every time you wish to run a simulation

```bash
cd ~/catkin_ws
source devel/setup.bash
export GAZEBO_MODEL_PATH=~/catkin_ws/src/ros_kortex/kortex_gazebo/models
```
At the end of installation and set-up, your file structure should look like this:
```
- catkin_ws
    - build
    - devel
    - src
        - jaco-gym
        - rlkit
        - ros_kortex
        - ros_kortex_vision
        - ros_numpy
        - ...
```
## Test your environment

### For the physical arm

In terminal 1, launch the robot connection:
```bash
roslaunch kortex_driver kortex_driver.launch dof:=6 gripper:=robotiq_2f_85 ip_address:=192.168.1.10
```

In terminal 2, run the python script to test robot actions:
```bash
python3 src/jaco-gym/scripts/test_actions.py
```

### For the arm in Gazebo (tested on ROS Melodic and Kinetic)

In terminal 1, cd into the ros-kortex directory and launch the simulated robot environment:
```bash
roslaunch kortex_gazebo spawn_kortex_robot.launch dof:=6 gripper:=robotiq_2f_85 start_rviz:=false
```

In terminal 2, run ros_kortex vision to publish camera data:
```bash
roslaunch kinova_vision kinova_vision.launch
```

In terminal 3, run the python script to test robot actions:
```bash
python3 src/jaco-gym/scripts/test_actions.py
```

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


## Environment details

### Observation

If reading the full state:

Type: Box(36)    

| Num           | Observation                        | Min   | Max  |
| ------------- | ---------------------------------- | ----- | ---- |
| 0             | joint_1 angle (rad)                | -inf  | inf  |
| 1             | joint_2 angle (rad)                | -inf  | inf  |
| 2             | joint_3 angle (rad)                | -inf  | inf  |
| 3             | joint_4 angle (rad)                | -inf  | inf  |
| 4             | joint_5 angle (rad)                | -inf  | inf  |
| 5             | joint_6 angle (rad)                | -inf  | inf  |
| 6             | joint_finger_1 angle (rad)         | -inf  | inf  |
| 7             | joint_finger_2 angle (rad)         | -inf  | inf  |
| 8             | joint_finger_3 angle (rad)         | -inf  | inf  |
| 9             | joint_finger_tip_1 angle (rad)     | -inf  | inf  |
| 10            | joint_finger_tip_2 angle (rad)     | -inf  | inf  |
| 11            | joint_finger_tip_3 angle (rad)     | -inf  | inf  |
| 12            | joint_1 velocity (rad/s)           | -inf  | inf  |
| 13            | joint_2 velocity (rad/s)           | -inf  | inf  |
| 14            | joint_3 velocity (rad/s)           | -inf  | inf  |
| 15            | joint_4 velocity (rad/s)           | -inf  | inf  |
| 16            | joint_5 velocity (rad/s)           | -inf  | inf  |
| 17            | joint_6 velocity (rad/s)           | -inf  | inf  |
| 18            | joint_finger_1 velocity (rad/s)    | -inf  | inf  |
| 19            | joint_finger_2 velocity (rad/s)    | -inf  | inf  |
| 20            | joint_finger_3 velocity (rad/s)    | -inf  | inf  |
| 21            | joint_finger_tip_1 velocity (rad/s)| -inf  | inf  |
| 22            | joint_finger_tip_2 velocity (rad/s)| -inf  | inf  |
| 23            | joint_finger_tip_3 velocity (rad/s)| -inf  | inf  |
| 24            | joint_1 effort (N.m)               | -inf  | inf  |
| 25            | joint_2 effort (N.m)               | -inf  | inf  |
| 26            | joint_3 effort (N.m)               | -inf  | inf  |
| 27            | joint_4 effort (N.m)               | -inf  | inf  |
| 28            | joint_5 effort (N.m)               | -inf  | inf  |
| 29            | joint_6 effort (N.m)               | -inf  | inf  |
| 30            | joint_finger_1 effort (N.m)        | -inf  | inf  |
| 31            | joint_finger_2 effort (N.m)        | -inf  | inf  |
| 32            | joint_finger_3 effort (N.m)        | -inf  | inf  |
| 33            | joint_finger_tip_1 effort (N.m)    | -inf  | inf  |
| 34            | joint_finger_tip_2 effort (N.m)    | -inf  | inf  |
| 35            | joint_finger_tip_3 effort (N.m)    | -inf  | inf  |


If reading the simplified state:

Type: Box(12)    


| Num           | Observation                        | Min   | Max  |
| ------------- | ---------------------------------- | ----- | ---- |
| 0             | joint_1 angle (rad)                | -inf  | inf  |
| 1             | joint_2 angle (rad)                | -inf  | inf  |
| 2             | joint_3 angle (rad)                | -inf  | inf  |
| 3             | joint_4 angle (rad)                | -inf  | inf  |
| 4             | joint_5 angle (rad)                | -inf  | inf  |
| 5             | joint_6 angle (rad)                | -inf  | inf  |
| 6            | joint_1 velocity (rad/s)            | -inf  | inf  |
| 7            | joint_2 velocity (rad/s)            | -inf  | inf  |
| 8            | joint_3 velocity (rad/s)            | -inf  | inf  |
| 9            | joint_4 velocity (rad/s)            | -inf  | inf  |
| 10            | joint_5 velocity (rad/s)           | -inf  | inf  |
| 11            | joint_6 velocity (rad/s)           | -inf  | inf  |



### Actions
Type: Box(6)

| Num           | Action                        | Min   | Max  |
| ------------- | ----------------------------- | ----- | ---- |
| 0             | joint_1 angle (scaled)        | -1    | 1    |
| 1             | joint_2 angle (scaled)        | -1    | 1    |
| 2             | joint_3 angle (scaled)        | -1    | 1    |
| 3             | joint_4 angle (scaled)        | -1    | 1    |
| 4             | joint_5 angle (scaled)        | -1    | 1    |
| 5             | joint_6 angle (scaled)        | -1    | 1    |

Note, at the moment joint_2 angle is restricted to 180 deg and joint_3 angle is restricted to the interval [90, 270] deg in order to reduce the arm's amplitude of motion.

### Reward
The reward is incremented at each time step by the negative of the distance between the target object position and the end deflector position (joint_6).


### Starting State
The arm is initialised with its joint angles as follows (in degrees): [0, 180, 180, 0, 0, 0].
The target object is initialised to a random location within the arm's reach.


### Episode Termination
An episode terminates if more than 50 time steps are completed.


### Step info
The info dictionary returned by the env.step function is structured as follows:
```python
info = {'tip coordinates': [x, y, z], 'target coordinates': array([x, y, z])}
```

## Python profiling
You can profile the time individual lines of code take to execute to monitor the code performance using [line_profiler](https://github.com/rkern/line_profiler).

### Install line-profiler

```bash
pip install line-profiler
```

### Decorate the functions you want to profile with @profile

For example:

```bash
vim scripts/0_test_jaco_gazebo_action_gym.py
```

```python
@profile
def main():

    for episode in range(3):

        obs = env.reset()
        ...
```

### Execute code and profile

```bash
kernprof -l 0_test_jaco_gazebo_action_gym.py
```

### Read profiling results line by line

```bash
python -m line_profiler 0_test_jaco_gazebo_action_gym.py.lprof > profiling_result_test.txt
```

## Supported systems
Tested on:
- Ubuntu 18.04 and 16.04 
- Python 3.6.9
- Gym 0.15.4
