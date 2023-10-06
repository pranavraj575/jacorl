* install ROS Noetic (tested on Noetic and Melodic)
* source the install, need to do this in every terminal or add to bash rc
    ```{bash}
  source /opt/ros/noetic/setup.bash
    ```
*  install packages

    ```{bash}
    sudo apt-get update && sudo apt-get install -y cmake libopenmpi-dev python3-dev zlib1g-dev python3-pip swig ffmpeg
    ```

    ```{bash}
    sudo pip3 install rospkg catkin_pkg
    ```
    
* install version specific packages (replace `<dist>` with distribution (i.e. `noetic`))
  
    ```{bash}
   sudo apt-get install -y ros-<dist>-gazebo-ros-control
    ```

    ```{bash}
    sudo apt-get install -y ros-<dist>-ros-controllers*
    sudo apt-get install -y ros-<dist>-trac-ik-kinematics-plugin
    sudo apt-get install -y ros-<dist>-effort-controllers 
    sudo apt-get install -y ros-<dist>-joint-state-controller 
    sudo apt-get install -y ros-<dist>-joint-trajectory-controller 
    sudo apt-get install -y ros-<dist>-controller-*
    sudo apt-get install -y ros-<dist>-rgbd-launch
    ```
* make project directory (run in whereever you want the project)
    ```{bash}
    mkdir -p ~/catkin_ws
    cd ~/catkin_ws
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
    TODO: check if installing ros kortex is necessary
* install the correct ros kortex (link) into the folder
  ```{bash}
  git clone -b <branch-name> https://github.com/Kinovarobotics/ros_kortex.git
  ```
* TODO: copy the relevant files over

  need models in the same place

  need to source devel
  
* ros_kortex/kortex_gazebo/worlds/jaco.world
* ros_kortex/kortex_gazebo/launch/*

* Do this too
    ```{bash}
    sudo apt install -y gstreamer1.0-tools gstreamer1.0-libav libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev gstreamer1.0-plugins-good gstreamer1.0-plugins-base
   ```
* Build
   ```{bash}
   sudo python3 -m pip install conan==1.59
   conan config set general.revisions_enabled=1
   conan profile new default --detect > /dev/null
   conan profile update settings.compiler.libcxx=libstdc++11 default   
   ```
   ```{bash}
   sudo apt install -y python3-rosdep2
   rosdep update 
   rosdep install --from-paths src --ignore-src -y --rosdistro <dist>
   ```
   ```{bash}
   cd ~/catkin_ws
   sudo apt install -y python3-catkin-tools python3-osrf-pycommon
   catkin init
   catkin_make 
   ```
  

  
