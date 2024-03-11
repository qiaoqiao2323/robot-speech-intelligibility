###################################################################################################################
##################################################### Installation ################################################
###################################################################################################################

To install : Python 2.7 (https://www.python.org/downloads/)
	     kivy library (https://kivy.org/#download)
	     naoqi library


Open the file "data.txt" if you have chosen to use a program with a robot.
Turn on the robot with a press for a few seconds on the button located on its torso.
Press for a short time on this button when the robot is turned on and the robot is going to tell you its IP address.
Modify the IP address in the file by replacing it with the IP address given by the robot.
The IP address should be like : 192.168.1.13
Save the "data.txt" file.

Run the main.py

A window will appear on the screen requesting the ID of the participant.
The participant ID permits to save different observations collected during the word recognition test in a file text proper to 
the participant.
Two participants can't have the same ID.

To collect each information, close the window by just using the button "close the BART test".
The button "close the test" appears in the end window when the test is over.
To show the end window, you can use two methods :
	1.Finish the test after using 50 words.
	2.Use the "quit" button and after push the "yes" button to confirm the end of the experiment.

If another  method is used to close the test, some data will be lost (points earned, time in millisecond between
two clicks, the time in seconds on the game and the time of the end of the experiment).

To access at results files, go to the "word_recognition" folder and after in the "results" folder.
Names of results files are in the shape of "results_participantID.txt".

###################################################################################################################
############################################## aim of the test ####################################################
###################################################################################################################

Rules of the game:
We are going to have a word recognition game.
one after the other.
The robot will randomly select a word and pronounce each letter one at a time.
You need to type in your answer,
and your pleasantness about the robot voice from 1 to 10. 
You can only submit one answer per word. 
Each correct letter earns you a point,
and incorrect or incomplete answers receive no points,
You can click "Submit" button to submit your answer upon you finish it.
and start with a new word. Remember,
you have 50 words in total.
 Attention: if you quit the test before the end, you will not earn anything.
If you have questions, it is the right time to ask! Good luck!  '''







