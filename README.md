# bin_picking
Bin Picking research project 

Bin Picking avec Deep Learning, voir [ici](https://www.youtube.com/watch?v=ydh_AdWZflA)

## Requirements

1. Follow [ROS Tutorials](http://wiki.ros.org/ROS/Tutorials) to have a better understanding of how ROS works
2. Follow [Universal Robot Tutorial](https://academy.universal-robots.com/es/formacion-online/formacion-online-de-cb3/)
3. Create your catkin_workspace: 
4. Install the following:
   `cd catkin_ws/src`
   `git clone https://github.com/UniversalRobots/Universal_Robots_ROS_Driver`

   `cd ..`
   `git clone -b calibration_devel https://github.com/fmauch/universal_robot.git src/fmauch_universal_robot`

   `sudo apt update -qq`
   `rosdep update`
   `rosdep install --from-paths src --ignore-src -y`

   `git clone https://github.com/ros-controls/ros_controllers.git  src/ros_controllers`

   `sudo apt-get install ros-melodic-four-wheel-steering-controller`

   `git clone https://github.com/raiv-toulouse/ur_icam.git src/ur_icam`

   `catkin_make`
   `ls -l`

5. Camera calibration. The first time you use this driver, you must extract the calibration from the robot to a file. (IP of the robot)
   - `roslaunch ur_calibration calibration_correction.launch robot_ip:=10.31.56.102 target_filename:="${HOME}/Calibration/ur3_calibration.yaml"`

6. In the UR Robot, configure the IP of the computer for the communication. 

Note: dont forget to configure ROS_MASTER in every computer 

## Environment Setup 
Note: Please enter the each of the following commands in a new terminal

1. Initialize ROS, Moveit and Universal Robot. Must be run in the same computer

- `roslaunch ur_robot_driver ur3_bringup.launch robot_ip:=10.31.56.102 kinematics_config:=${HOME}/Calibration/ur3_calibration.yaml`

On the robot, a program with this instruction has been saved under the name of 'communication_with_ros.urp')
By clicking on the Command tab after having selected the ExternalControl command, make sure that the Host IP is correct (IP of the computer on which the previous roslaunch command was launched).
If this is not the case, click on the Installation tab then External Control to correct this. 

Finally, you have to press the small Play button at the bottom of the graphical interface. 
If you now return to the roslaunch terminal, the following lines should have appeared: 


- `roslaunch ur3_moveit_config ur3_moveit_planning_execution.launch`

2. Initialize the Robot Controller. Same computer as before
- `rosrun robot_controller main.py` 

3. Raspberry Pi. Initialize the camera, and the information from arduino. 
- `roslaunch usb_cam usb_cam-test.launch`
- `rosrun rosserial_arduino serial_node.py _port:=/dev/ttyACM0/`

4. Initialize the other camera. It can run in any computer
- `usb_cam usb_cam-test.launch`

5. Run AI Manager (Use it in the DL Server)
- `rosrun ai_manager main.py`






