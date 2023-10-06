# Kinova Jaco gen3 RL Environment for 2 finger
This repository contains a complete environment for Reinforcement Learning that can be run both in simulation and on a physical Kinova arm. We provide the base robot environment under ```jaco-gym/jaco_gym/envs/robot_env.py``` and also provide an example task environment under ```jaco-gym/jaco_gym/envs/task_envs/stack_cups_gazebo.py``` that facilitates training for a cup-stacking robot (shown below). Additonally, this repo contains implementations of numerous RL training methods that can be used for your learning task under the ```rlkit``` submodule.


![Jaco Gazebo](/jaco_world.png)

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

WARNING: THIS DOES NOT WORK WITH CONDA! deactivate it or uninstall it first

#### 1. Install [ROS](http://wiki.ros.org/ROS/Installation) (if it is not already installed)

* ROS Melodic on Ubuntu 18.04
* ROS Noetic on Ubuntu 20.04

Make sure the following line is in the .bashrc file (with `<distro>` replaced by ROS distribution, e.g. `noetic`) or run every time we open a terminal

```bash
source /opt/ros/<distro>/setup.bash
```

Then complete the following sudo installs:

```bash
sudo apt-get update && sudo apt-get install -y cmake libopenmpi-dev python3-dev zlib1g-dev python3-pip swig ffmpeg python3-catkin-tools python3-osrf-pycommon
sudo pip3 install rospkg catkin_pkg
```

#### 2. Install the robot controllers

```bash
sudo apt-get install -y ros-<distro>-gazebo-ros-control
sudo apt-get install -y ros-<distro>-ros-controllers*
sudo apt-get install -y ros-<distro>-trac-ik-kinematics-plugin
sudo apt-get install -y ros-<distro>-effort-controllers 
sudo apt-get install -y ros-<distro>-joint-state-controller 
sudo apt-get install -y ros-<distro>-joint-trajectory-controller 
sudo apt-get install -y ros-<distro>-controller-*
sudo apt-get install -y ros-<dist>-rgbd-launch
```
(replace `<distro>` by your ROS distribution, for example `kinetic` or `melodic`)

#### 3. Install and configure your [Catkin workspace](http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment), with [jacorl](https://github.com/pranavraj575/jacorl) (this repository) as src
Since this repo contains sub-repositories, you will need to cd into some of the subdirectories and run pip3 install -e . to properly install the required packages (see / copy the lines below):

```bash
mkdir -p ~/catkin_ws
cd ~/catkin_ws
git clone https://github.com/pranavraj575/jacorl src
cd src/jaco-gym
pip3 install -e .
cd ../rlkit
pip3 install -e .
cd ../ros_numpy
pip3 install -e .
```

Note: if the `jaco-gym` install doesnt work, try giving it one more go before panicking

#### 4. Clone the correct branch of ros_kortex, and copy files over (`<branch-name>` should look like `noetic-devel`)
```bash
cd ~/catkin_ws/src
git clone -b <branch-name> https://github.com/Kinovarobotics/ros_kortex.git
cp -r setup/kortex_gazebo/* ros_kortex/kortex_gazebo
cp setup/misc/gen3_macro.xacro ros_kortex/kortex_description/arms/gen3/6dof/urdf/
```
#### 5. Install dependencies for the ros_kortex and ros_kortex_vision package. For more detailed instructions, see [here](https://github.com/Kinovarobotics/ros_kortex) and [here](https://github.com/pranavraj575/jacorl/tree/master/ros_kortex_vision).
```bash
sudo apt install -y gstreamer1.0-tools gstreamer1.0-libav libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev gstreamer1.0-plugins-good gstreamer1.0-plugins-base
sudo apt install -y python3-rosdep2
``` 

```bash
sudo python3 -m pip install conan==1.59
conan config set general.revisions_enabled=1
conan profile new default --detect > /dev/null
conan profile update settings.compiler.libcxx=libstdc++11 default
cd ~/catkin_ws
rosdep update
rosdep install --from-paths src --ignore-src -y --rosdistro <distro>
```

(replace `<branch-name>` and `<distro>` with your corresponding ROS distribution, for example `kinetic` or `melodic`)


#### 6. Install the ROS packages and build.
* ```bash
  cd ~/catkin_ws
  catkin clean
  catkin init
  catkin_make
  ```


Note, the kinova-ros package was adapted from the [official package](https://github.com/Kinovarobotics/kinova-ros).

#### 7. Add the following lines to your bashrc file, or run them every time you open a terminal

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
roslaunch kortex_driver kortex_driver.launch dof:=6 gripper:=robotiq_2f_85 ip_address:=192.168.1.10 start_rviz:=false
```

In terminal 2, run ros_kortex vision to publish camera data:
```bash
roslaunch kinova_vision kinova_vision.launch
```

In terminal 3, run the python script to test robot actions:
```bash
python3 src/jaco-gym/scripts/test_actions.py
```

### For the arm in Gazebo (tested on ROS Melodic and Kinetic)

In terminal 1, cd into the ros-kortex directory and launch the simulated robot environment:
```bash
roslaunch kortex_gazebo spawn_kortex_robot.launch dof:=6 gripper:=robotiq_2f_85 start_rviz:=false
```

In terminal 2, run the python script to test robot actions:
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

## Supported systems
Tested on:
- Ubuntu 18.04; ROS Melodic; Python 3.6.9
- Ubuntu 20.04; ROS Noetic; Python 3.8.10
