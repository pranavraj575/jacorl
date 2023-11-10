## Installation

Note: all commands to run for installation should be on this page, with the exception of [step 1 (installing ROS)](https://github.com/pranavraj575/jacorl/blob/master/README.md#1-install-ros-if-it-is-not-already-installed). The hyperlinks are just for more information/if any errors occur

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
sudo apt-get install -y ros-<distro>-rgbd-launch
```
(replace `<distro>` by your ROS distribution, for example `noetic` or `melodic`)

#### 3. Install and configure your [Catkin workspace](http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment), with [jacorl](https://github.com/pranavraj575/jacorl) (this repository) as src
Since this repo contains sub-repositories, you will need to cd into some of the subdirectories and run pip3 install -e . to properly install the required packages (see / copy the lines below):

```bash
mkdir -p ~/catkin_ws
cd ~/catkin_ws
git clone https://github.com/pranavraj575/jacorl src
pip3 install -e src/jaco-gym/
pip3 install -e src/rlkit/
pip3 install -e src/ros_numpy/
```

Note: if the `jaco-gym` install doesnt work, try giving it one more go before panicking

#### 4. Clone the correct branch of [ros_kortex](https://github.com/Kinovarobotics/ros_kortex), and copy files over (`<branch-name>` should look like `noetic-devel`)
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

(replace `<branch-name>` and `<distro>` with your corresponding ROS distribution, for example `noetic` or `melodic`)


#### 6. Install the ROS packages and build.
* ```bash
  cd ~/catkin_ws
  catkin init
  catkin_make
  ```


Note, the kinova-ros package was adapted from the [official package](https://github.com/Kinovarobotics/kinova-ros).

#### 7. Add the following lines to your bashrc file, or run them every time you open a terminal

```bash
cd ~/catkin_ws
source devel/setup.bash
export GAZEBO_MODEL_PATH=~/catkin_ws/src/models:$GAZEBO_MODEL_PATH
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
#### 8. For conneecting to the physical arm:

Change your laptop IP to 192.168.1.11 or 192.168.2.11 (not sure what the logic here is)

Also, set subnet mask to 255.255.255.0

Test connection by eitherr going to http://192.168.1.10 or going to the robot ip

Username/password are both admin

## Test your environment

### For the physical arm

In terminal 1, launch the robot connection:
```bash
roslaunch kortex_driver kortex_driver.launch dof:=6 gripper:=robotiq_2f_85 ip_address:=192.168.1.10 start_rviz:=false
```

In terminal 2, run ros_kortex vision to publish camera data:
```bash
roslaunch kinova_vision robot_eye.launch device:=192.168.1.10
```

In terminal 3, run the python script to test robot actions:
```bash
python3 src/jaco-gym/scripts/test_actions.py
```

### For the arm in Gazebo (tested on ROS Melodic and Noetic)

In terminal 1, cd into the ros-kortex directory and launch the simulated robot environment:
```bash
roslaunch kortex_gazebo spawn_kortex_robot.launch dof:=6 gripper:=robotiq_2f_85 start_rviz:=false
```

In terminal 2, run the python script to test robot actions:
```bash
python3 src/jaco-gym/scripts/test_actions.py
```

