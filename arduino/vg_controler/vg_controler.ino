/* 
 * Vacuum Gripper Controler
 * - power On/Off the Vacuum Gripper (VG)
 * - get the distance between VG and the objects on the ground
 * - get information about if an object is gripped
 */

#include <ros.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>

ros::NodeHandle  nh;
std_msgs::Bool object_gripped_msg;
std_msgs::Float32 distance_msg;

const int on_off_pin = 12;  // Used to switch on/off the Vacuum Gripper
const int object_gripped_pin = 11;  // Used to read if the Vacuum Gripper has got an object
const int distance_pin = 0;   // Used to read the distance between Vacuum Gripper and objects

void onOffMsgCb( const std_msgs::Empty& toggle_msg){
  digitalWrite(on_off_pin, HIGH-digitalRead(on_off_pin));
}

ros::Subscriber<std_msgs::Empty> sub_on_off("switch_on_off", &onOffMsgCb );
ros::Publisher pub_object_gripped( "object_gripped", &object_gripped_msg);
ros::Publisher pub_distance( "distance", &distance_msg);

void setup()
{ 
  pinMode(on_off_pin, OUTPUT);
  nh.initNode();
  nh.subscribe(sub_on_off);
  nh.advertise(pub_object_gripped);
  nh.advertise(pub_distance);
}

void loop()
{  
  object_gripped_msg.data = digitalRead(object_gripped_pin);
  distance_msg.data = analogRead(distance_pin);
  nh.spinOnce();
  delay(1);
}
