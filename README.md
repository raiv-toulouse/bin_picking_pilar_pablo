# bin_picking
Bin Picking research project 

Bin Picking avec Deep Learning, voir [ici](https://www.youtube.com/watch?v=ydh_AdWZflA)

## Environment Setup 

1. roslaunch ur_robot_driver ur3_bringup.launch robot_ip:=10.31.56.102 kinematics_config:=${HOME}/Calibration/ur3_calibration.yaml (Pilar)

2. roslaunch ur3_moveit_config ur3_moveit_planning_execution.launch (Pilar)

3. Initialize the Robot Controller
   - `rosrun robot_controller main.py` (Pilar)

4.Raspberry Pi: `roslaunch usb_cam usb_cam-test.launch`
5. Raspberry Pi: `rosrun rosserial_arduino serial_node.py _port:=/dev/ttyACM0/`
6. roslaunch `usb_cam usb_cam-test.launch` (Raspberry Pi)
7. roslaunch `usb_cam usb_cam-test.launch` (Pablo)
8. DL: `rosrun ai_manager main.py`





