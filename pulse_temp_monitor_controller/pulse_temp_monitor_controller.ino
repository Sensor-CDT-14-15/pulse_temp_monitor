#include "Timer.h"

const int RED_LED_PIN=13;
const int IR_LED_PIN=12;
const int RED_DIODE_PIN=A1;
const int IR_DIODE_PIN=A2;
const int FINGER_TEMP_PIN=A0;
const int ROOM_TEMP_PIN=A3;

const int READ_TIME=20;

float tempFinger;
float tempRoom;

Timer readTimer;

void setup() {
  pinMode(RED_LED_PIN, OUTPUT);
  pinMode(IR_LED_PIN, OUTPUT);
  digitalWrite(RED_LED_PIN, HIGH);
  digitalWrite(IR_LED_PIN, LOW);
  Serial.begin(9600);

  readTimer.every(READ_TIME, echoReadings);
}

void loop() {
  readTimer.update();
}

void echoReadings() {
  Serial.print(analogRead(RED_DIODE_PIN));
  Serial.print("\t");
  tempFinger = analogRead(FINGER_TEMP_PIN) * 50.0 / 1023;
  Serial.print(tempFinger);
  Serial.print("\t");
  Serial.print(analogRead(IR_DIODE_PIN));
  Serial.print("\t");
  tempRoom = analogRead(ROOM_TEMP_PIN) * 50.0 / 1023;
  Serial.println(tempRoom);
}

