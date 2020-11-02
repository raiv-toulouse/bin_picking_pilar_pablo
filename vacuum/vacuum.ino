const int pin_distance = 8;
const int pin_grasp = 10;
const int pin_on_off = 12;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(pin_on_off, OUTPUT);
  pinMode(pin_distance, INPUT);
  pinMode(pin_grasp, INPUT);
  digitalWrite(pin_on_off, HIGH);
}

void releaseObject()
{
  Serial.println("Contact released in 5 s");
  delay(5000);
  digitalWrite(pin_on_off, LOW);
  delay(500);
  digitalWrite(pin_on_off, HIGH);
}

void loop() {
  // put your main code here, to run repeatedly:
  int grasp = digitalRead(pin_grasp);
  //boolean grasp = (digitalRead(pin_grasp) == LOW);
  Serial.println(grasp);
  if (grasp)
    //releaseObject();
    Serial.println("grasp");
  else
    Serial.println("No grasp");
  boolean near = (digitalRead(pin_distance) == LOW);
  if (near)
    Serial.println("Near ");
  else
    Serial.println("Far");
  Serial.println("");
  delay(1000);
}
