

/* 
 * Vacuum Gripper Controler
 * - power On/Off the Vacuum Gripper (VG)
 * - get information about if contact between VG and the objects on the ground
 * - get information about if an object is gripped
 */

#include <ros.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>

ros::NodeHandle  nh;
std_msgs::Bool object_gripped_msg;
std_msgs::Bool contact_msg;

const int on_off_pin = 12;  // Used to switch on/off the Vacuum Gripper
const int object_gripped_pin = 10;  // Used to read if the Vacuum Gripper has got an object
//const int distance_pin = 8;   // Used to read the distance between Vacuum Gripper and objects
const int contact_pin = 6;   // Used to read the distance between Vacuum Gripper and objects

boolean contact;

void onOffMsgCb( const std_msgs::Bool& toggle_msg);

ros::Subscriber<std_msgs::Bool> sub_on_off("switch_on_off", &onOffMsgCb );
ros::Publisher pub_object_gripped( "object_gripped", &object_gripped_msg);
ros::Publisher pub_contact( "contact", &contact_msg);

void onOffMsgCb( const std_msgs::Bool& toggle_msg){
  if (toggle_msg.data){
    digitalWrite(on_off_pin, HIGH);
    digitalWrite(LED_BUILTIN, HIGH);
  } else {
    digitalWrite(on_off_pin, LOW);
    digitalWrite(LED_BUILTIN, LOW);
  }
}

void setup()
{ 
  Serial.begin(57600);
  Serial.println("Serial prepared");
  
  nh.initNode();
  nh.subscribe(sub_on_off);
  nh.advertise(pub_object_gripped);
  nh.advertise(pub_contact);
  
  pinMode(on_off_pin, OUTPUT);
  pinMode(contact_pin, INPUT);
  pinMode(object_gripped_pin, INPUT);
  randomSeed(analogRead(1));
}

void loop()
{
  boolean object_gripped = (digitalRead(object_gripped_pin) == HIGH);
  object_gripped_msg.data = object_gripped;
  pub_object_gripped.publish(&object_gripped_msg);
  
  contact = (digitalRead(contact_pin) == HIGH);
  contact_msg.data = contact;
  pub_contact.publish(&contact_msg);
  
  nh.spinOnce();

  delay(50);
}
