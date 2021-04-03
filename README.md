# Automated-Society-Security-Task

## Problem Statement
* Automation of Society Security Tasks – Visitors in a housing society fall in the category of regular visitors lik edrivers, maids, milkman, sweeper, car cleaners, etc who visit on a daily basis and the other category is of infrequent known or unknown visitors ( like guests, delivery boys).
* Record with timestamp registered regular visitors and their temperature check.

## Problem Dimensions:
This problem has the following dimension
* First time registration of Regular visitors as described above.
* All subsequent visits, visitors data (facial recognition) should be captured using a continuous camera feed along with their body temperature also recorded in the solution.
* The facial recognition should take care of visual changes in appearance of a person say with or without beard/moustache, glasses, raincoat or even helmet.
* The solution should also present a end of day report.
* Temperature reading of the visitor also needs to be done. This can be done either with a separate device or a single device for camera and temperature both. However please note that an integrated solution should be presented with no manual intervention for either temperature / face recording.

## Hardware:
1. Raspberry pi 4b
2. Pi Camera
3. MLX90614 for Contactless Temperature Sensor.

## Softwares/Interfaces:
1. OS: RaspbianOS (Linux type)
2. Python in jupyter notebook
3. Method: Convolutional Neural Networks
4. Packages for Face and Mask Recogniton : Opencv, Tensorflow, Keras, pyzbar, face_recognition
5. Tkinter for local GUI, PHP for webapp.
6. MySQL for database

## Key Features
1. Completely Contactless System. 
2. Facial Identification both while wearing and not wearing a mask meaning a person does'nt need to remove his/her mask to get identified.
3. Temperature Checking.
4. Live website to monitor and display real time information about visitors.
5. Standalone system and along with local cache.
6. Data of both valid and invalid visits stored.
7. Regular cleaning of both cache data and visitor photos for storage management.

## Credits
* Face Recognition by [Behic Guven](https://towardsdatascience.com/building-a-face-recognizer-in-python-7fd6630c6340)
* Face Mask Detection by [balajisrinivas](https://github.com/balajisrinivas/Face-Mask-Detection)