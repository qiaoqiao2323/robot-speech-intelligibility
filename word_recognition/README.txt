###################################################################################################################
##################################################### Installation ################################################
###################################################################################################################

To install : Python 2.7 (https://www.python.org/downloads/)
	     kivy library (https://kivy.org/#download)
	     naoqi library


For tests with one balloon, all balloons will explode between 1 and 127 pumps according to a random function with 
an uniform distribution.
For tests with three balloon, red balloons will explode between 1 and 40 pumps, green balloons between 35 and 80 
pumps and finally blue balloons between 80 and 127 pumps.

Open the file "data.txt" if you have chosen to use a program with a robot.
Turn on the robot with a press for a few seconds on the button located on its torso.
Press for a short time on this button when the robot is turned on and the robot is going to tell you its IP address.
Modify the IP address in the file by replacing it with the IP address given by the robot.
The IP address should be like : 192.168.1.13
Save the "data.txt" file.

Run the 

A window will appear on the screen requesting the ID of the participant.
The participant ID permits to save different observations collected during the word recognition test in a file text proper to 
the participant.
Two participants can't have the same ID.

To collect each information, close the window by just using the button "close the BART test".
The button "close the BART test" appears in the end window when the test is over.
To show the end window, you can use two methods :
	1.Finish the test after using 30 balloons
	2.Use the "quit" button and after push the "yes" button to confirm the end of the experiment.

If another  method is used to close the BART test, some data will be lost (pound earned, time in millisecond between
two clicks, the time in seconds on the game and the time of the end of the experiment).

To access at results files, go in the "BART" folder and after in the "results" folder.
Names of results files are in the shape of "results_participantID.txt".

###################################################################################################################
############################################## aim of the test ####################################################
###################################################################################################################

This test is like a normal BART test.
It will be asked to the participant to blow up 30 balloons, one after another, each pump will provide more money if
the balloon doesn't explode before the participant uses the "collect reward button". 
The robot reacts at each collect of money or when the balloon explodes. The robot will react also if the participant
wants to quit the test before the end of the experiment by using the "quit" button (if the participant decides to 
quit the test before the end then he will earn noting).
Apart from these interventions, the robot will react for six balloons at most during the test.
If the robot is in careful mod, it will ask the participant, for balloons chosen with a random function, to stop to 
blow up the balloon after five pums, it will repeat its requests with different sentences until the user uses the 
"collect reward" button or until the explosion of the balloon.
If the robot is in imprudent mode, for six balloons chosen with a random function, when the participant will push 
the "collect reward" button the robot will ask to continue the pumping of the balloon with different sentences until
the participant pushes again the button "collect reward" or until the explosion of the balloon.









