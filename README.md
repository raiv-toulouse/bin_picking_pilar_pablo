# bin_picking
Bin Picking research project.
Bin Picking avec Deep Learning, voir [ici](https://www.youtube.com/watch?v=ydh_AdWZflA)

## Project Description
The objective of the project is to implement a Pick and Place task with a robotic arm using Artificial Intelligence techniques such as Image Recognition and Reinforcement Learning.  

The objects in the environment are distributed in a box randomly so the system will need to identify the objects and find an optimal set of movements for completing the task.  


## Architecture

![Architecture](/readme-images/ROS_architecture.png)

ROS will be the tool used to interact and communicate with the robot. 

- **AI Manager** Package containing all the Artificial Intelligence code. 
- **Robot Controller** - Package for controlling, moving and interacting with the robot. 
- **Arduino** - Code used for detection, already load in the arduino. Included for future modifications and improvements
- **Raspberry** - Usb_cam package included
- **Universal Robot** - UR robot used in the proyect

You can find a more detailed explanation inside each folder.

Disclaimer: This project works with **Ubuntu 18** and **ROS Melodic**. It's probable that for some parts of the project, different versions of python might be needed. 
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

6. Raspberry Installation:
- Install ROS and a catkin woskpace in Raspberry
- Install Arduino  and import ROS serial library for the Arduino connected to the Raspberry
- Install the [usb_camera package](https://github.com/ros-drivers/usb_cam)
   - Clone the repository in src
   - `catkin_make`  
   - `source devel/setup.bash`
- Find the arduino port (probably thie port is: /dev/ttyACM0/)

7. Don't forget to configure **ROS_MASTER** and **ROS_IP** in every computer and in the Raspberry to connect all the nodes in the architecture
   -`export ROS_MASTER=10.31.56.80` (IP Of the ROS_MASTER)
   -`export ROS_IP=10.31.56.75` (IP of nodes)

8. Install **Conda** to work with environments. Due to ROS packages and versions sometimes python2 or python3 will be needed. You can create and replicate the environments from folder **conda_environments** with command `conda env create -f name_of_file.yml`
  


## Environment Setup 
**Note**: Please enter the each of the following commands in a new terminal

1. Initialize ROS, Moveit and Universal Robot. Must be run in the same computer
- `roslaunch ur_robot_driver ur3_bringup.launch robot_ip:=10.31.56.102 kinematics_config:=${HOME}/Calibration/ur3_calibration.yaml`

On the robot, a program with this instruction has been saved under the name of 'communication_with_ros.urp')
By clicking on the Command tab after having selected the ExternalControl command, make sure that the Host IP is correct (IP of the computer on which the previous roslaunch command was launched).
If this is not the case, click on the Installation tab then External Control to correct this. 

Finally, you have to press the small Play button at the bottom of the graphical interface. 
If you now return to the roslaunch terminal, the following lines should have appeared: 
 *Robot requested program*  
 *Sent program to robot*  
 *Robot ready to receive control commands*  

- `roslaunch ur3_moveit_config ur3_moveit_planning_execution.launch`

2. Initialize the **Robot Controller**. Same computer as before (Python2)
- `conda activate python2` (if you are using conda environments)  
- `rosrun robot_controller main.py`  
   You should see the robot moving to the place position and coming back to the center. If not, probably something went worng in the previous steps. 

3. **Raspberry Pi** Initialize the camera, and the information from arduino. You can connect with the raspberry via ssh or with the screen and a keyboard.
-  Connection via ssh: `ssh pi@10.31.56.X`, (X can vary) Ex: Put the password: 123456
- `roslaunch usb_cam usb_cam-test.launch`
- `rosrun rosserial_arduino serial_node.py _port:=/dev/ttyACM0/`
   Don't forget to check if the IP changed (Keep the ROS_MASTER and the ROS_IP up to date) (Maybe you can use the screen and the keyboard to check the IP if ssh is not working)

4. Initialize the **other camera**. It can run in any computer
- `usb_cam usb_cam-test.launch`

5. Run **AI Manager** (Use it in the DL Server) (Python3)
- SSH to access the DL (or any remote computer) `ssh user@10.31.56.62` Put the password. Remember the IPs can change 
- `conda activate python3` (if you are using conda environments)  
- `rosrun ai_manager main.py`
 If all the steps before were done correctly,the robot should start moving. Actions should be seen in the cmd. If it doesn't and robot controller is running, it maybe because is waiting for the image. Go back to step 3, launch again the cameras. On the other hand, if the robot suddenly stops moving  the most probable cause is that ROS serial node in Raspebrry Pi is not working correctly. Go back to step 3 and check everything. 

**Note**: when using **usb_cam** you should be able to see the images taken from the cameras. Both from the onboard camera on the robot, and the upper camera with the whole environment. 







