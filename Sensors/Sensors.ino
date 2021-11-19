

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
const int led1_pin = 2;
const int led2_pin = 3;
const int led3_pin = 4;
const int led4_pin = 5;

const int object_gripped_pin = 10;  // Used to read if the Vacuum Gripper has got an object
//const int distance_pin = 8;   // Used to read the distance between Vacuum Gripper and objects
const int contact_pin = 8;   // Used to read the distance between Vacuum Gripper and objects

boolean contact;

void onOffMsgCb( const std_msgs::Bool& toggle_msg);
void onOffMsgLed1( const std_msgs::Bool& toggle_msg);
void onOffMsgLed2( const std_msgs::Bool& toggle_msg);
void onOffMsgLed3( const std_msgs::Bool& toggle_msg);
void onOffMsgLed4( const std_msgs::Bool& toggle_msg);

ros::Subscriber<std_msgs::Bool> sub_on_off("switch_on_off", &onOffMsgCb );

ros::Subscriber<std_msgs::Bool> led1_on_off("led1_on_off", &onOffMsgLed1 );
ros::Subscriber<std_msgs::Bool> led2_on_off("led2_on_off", &onOffMsgLed2 );
ros::Subscriber<std_msgs::Bool> led3_on_off("led3_on_off", &onOffMsgLed3 );
ros::Subscriber<std_msgs::Bool> led4_on_off("led4_on_off", &onOffMsgLed4 );

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

void onOffMsgLed1( const std_msgs::Bool& toggle_msg){
  if (toggle_msg.data){
    digitalWrite(led1_pin, HIGH);
    digitalWrite(LED_BUILTIN, HIGH);
  } else {
    digitalWrite(led1_pin, LOW);
    digitalWrite(LED_BUILTIN, LOW);
  }
}

void onOffMsgLed2( const std_msgs::Bool& toggle_msg){
  if (toggle_msg.data){
    digitalWrite(led2_pin, HIGH);
    digitalWrite(LED_BUILTIN, HIGH);
  } else {
    digitalWrite(led2_pin, LOW);
    digitalWrite(LED_BUILTIN, LOW);
  }
}

void onOffMsgLed3( const std_msgs::Bool& toggle_msg){
  if (toggle_msg.data){
    digitalWrite(led3_pin, HIGH);
    digitalWrite(LED_BUILTIN, HIGH);
  } else {
    digitalWrite(led3_pin, LOW);
    digitalWrite(LED_BUILTIN, LOW);
  }
}

void onOffMsgLed4( const std_msgs::Bool& toggle_msg){
  if (toggle_msg.data){
    digitalWrite(led4_pin, HIGH);
    digitalWrite(LED_BUILTIN, HIGH);
  } else {
    digitalWrite(led4_pin, LOW);
    digitalWrite(LED_BUILTIN, LOW);
  }
}

void setup()
{
  Serial.begin(57600);
  // Serial.println("Serial prepared");

  nh.initNode();
  nh.subscribe(sub_on_off);

  nh.subscribe(led1_on_off);
  nh.subscribe(led2_on_off);
  nh.subscribe(led3_on_off);
  nh.subscribe(led4_on_off);

  nh.advertise(pub_object_gripped);
  nh.advertise(pub_contact);

  pinMode(on_off_pin, OUTPUT);

  pinMode(led1_pin, OUTPUT);
  pinMode(led2_pin, OUTPUT);
  pinMode(led3_pin, OUTPUT);
  pinMode(led4_pin, OUTPUT);

  pinMode(contact_pin, INPUT);
  pinMode(object_gripped_pin, INPUT);
  randomSeed(analogRead(1));
}

void loop()
{
  boolean object_gripped = (digitalRead(object_gripped_pin) == HIGH);
  object_gripped_msg.data = object_gripped;
  Serial.println(object_gripped_msg.data);
  pub_object_gripped.publish(&object_gripped_msg);

  contact = (digitalRead(contact_pin) == HIGH);
  contact_msg.data = contact;
  pub_contact.publish(&contact_msg);

  nh.spinOnce();

  delay(50);
}
