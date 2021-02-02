# bin_picking
Bin Picking research project.
Bin Picking avec Deep Learning, voir [ici](https://www.youtube.com/watch?v=ydh_AdWZflA)

## Project Description
The objective of the project is to implement a Pick and Place task with a robotic arm using Artificial Intelligence techniques such as Image Recognition and Reinforcement Learning.  

The objects in the environment are distributed in a box randomly so the system will need to identify the objects and find an optimal set of movements for completing the task.  


## Architecture

![Architecture](ai_manager/readme-images/ROS_architecture.png)

ROS will be the tool used to interact and communicate with the robot. 

- **AI Manager** Package containing all the Artificial Intelligence code. 
- **Robot Controller** - Package for controlling, moving and interacting with the robot. 
- **Arduino** - Code used for detection, already load in the arduino. Included for future modifications and improvements
- **Raspberry** - Usb_cam package included
- **Universal Robot** - UR robot used in the proyect

You can find a more detailed explanation inside each folder.

Disclaimer: This project works with Ubuntu 18.08 and ROS Melodic. It's probable that for some parts of the project, different versions of python might be needed. 
It is **very recommended** to work with conda environments. 


## Requirements

1. Follow [ROS Tutorials](http://wiki.ros.org/ROS/Tutorials) to have a better understanding of how ROS works
2. Follow [Universal Robot Tutorial](https://academy.universal-robots.com/es/formacion-online/formacion-online-de-cb3/)
3. Create your catkin_workspace
4. Install the following:
   `cd catkin_ws/src`
   
   `git clone https://github.com/UniversalRobots/Universal_Robots_ROS_Driver`

   `cd ..`
   
   `git clone -b calibration_devel https://github.com/fmauch/universal_robot.git src/fmauch_universal_robot`
   
   Edit the fmauch_universal_robot/ur3_moveit_config/config/controllers.yaml file and write this line (to correct this problem : https://github.com/ros-industrial/ur_modern_driver/issues/261#issuecomment-462254653): 
   
   `action_ns: scaled_pos_joint_traj_controller/follow_joint_trajectory`

   `sudo apt update -qq`
   
   `rosdep update`
   
   `rosdep install --from-paths src --ignore-src -y`

   `git clone https://github.com/ros-controls/ros_controllers.git  src/ros_controllers`

   `sudo apt-get install ros-melodic-four-wheel-steering-controller`

   `git clone https://github.com/raiv-toulouse/ur_icam.git src/ur_icam`

   `catkin_make`
   
   `conda activate python2`
   
   To correct a bug (Failed to import pyassimp, see https://github.com/ros-planning/moveit/issues/86 for more info)
   
   `~/anaconda3/envs/python2/bin/pip install  pyassimp`

5. Camera calibration. The first time you use this driver, you must extract the calibration from the robot to a file. (IP of the robot)
   - `roslaunch ur_calibration calibration_correction.launch robot_ip:=10.31.56.102 target_filename:="${HOME}/Calibration/ur3_calibration.yaml"`

6. Raspberry Installation:
- Install ROS and a catkin woskpace in Raspberry
- Install Arduino  and import ROS serial library for the Arduino connected to the Raspberry
- Install the [usb_camera package](https://github.com/ros-drivers/usb_cam)
- Find the arduino port (probably this step is not necessary)

7. Don't forget to configure ROS_MASTER and ROS_IP in every computer and in the Raspberry to connect all the nodes in the architecture
   -`export ROS_MASTER=10.31.56.80` (IP Of the ROS_MASTER)
   -`export ROS_IP=10.31.56.75` (IP of nodes)


## Environment Setup 
Note: Please enter the each of the following commands in a new terminal

1. Initialize ROS, Moveit and Universal Robot. Must be run in the same computer

- `roslaunch ur_robot_driver ur3_bringup.launch robot_ip:=10.31.56.102 kinematics_config:=${HOME}/Calibration/ur3_calibration.yaml`

On the robot, a program with this instruction has been saved under the name of 'communication_with_ros.urp')
By clicking on the Command tab after having selected the ExternalControl command, make sure that the Host IP is correct (IP of the computer on which the previous roslaunch command was launched).
If this is not the case, click on the Installation tab then External Control to correct this. 

Finally, **you have to press the small Play button at the bottom of the graphical interface**. 
If you now return to the roslaunch terminal, the following lines should have appeared: 
 *Robot requested program*  
 *Sent program to robot*  
 *Robot ready to receive control commands*  

- `roslaunch ur3_moveit_config ur3_moveit_planning_execution.launch`

2. Initialize the Robot Controller. Same computer as before but with a python2 environment.
- `rosrun robot_controller main.py` 

3. Raspberry Pi. Initialize the camera, and the information from arduino.
- `roslaunch usb_cam usb_cam-test.launch`
- `rosrun rosserial_arduino serial_node.py _port:=/dev/ttyACM0`

4. Initialize the other camera. It can run in any computer
- `usb_cam usb_cam-test.launch`

5. Run AI Manager (Use it in the DL Server)
- `rosrun ai_manager main.py`






