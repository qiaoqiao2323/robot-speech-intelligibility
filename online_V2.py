# -*- coding: cp1252 -*-
# "C:\Program Files (x86)\Python27\python" C:\Users\erohart\Desktop\BART\BART_objet_incite.py
from __future__ import division
import ast
import random
import re
import subprocess
import kivy  # import from the kivy part
from random import randint
from kivy.app import *
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import NumericProperty
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.core.audio import SoundLoader
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.uix.textinput import TextInput
from kivy.uix.progressbar import ProgressBar
from kivy.core.audio import SoundLoader
import phonetics
from kivy.core.window import Window
from functools import partial
import naoqi, sys, argparse, almath  # imports from the nao part
import motion
from naoqi import *
from optparse import OptionParser
import time, threading, thread
import os, inspect
from os.path import exists
import sys

absol = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0])) + "/data.txt"
f = open(absol, 'r')
# IP = f.readline() #IP adress of the robot
# chang playwav function to change annoyance level in the environment.
IP = "169.254.81.2"
# IP = "192.168.0.102"
#IP = "10.10.147.196"
# IP = "169.254.81.2"
#IP = "127.0.0.1"

# IP = "192.168.0.130"
f.close()
PORT = 9559  # port to conect the robot
path = os.path.realpath(
    os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0])) + "/results/results_"


class RobotControllerModule(ALModule):
    def __init__(self, name):
        ALModule.__init__(self, name)
        self.tts = ALProxy("ALTextToSpeech") # initialize speech
        self.motionProxy = ALProxy("ALMotion") # help make the robot move
        self.animatedSpeechProxy = ALProxy("ALAnimatedSpeech") # initialize the body language
        self.tracker = ALProxy("ALTracker")   # initialize the track of different targets
        self.autonomousLife = ALProxy("ALAutonomousLife") # initialize autonomous life
        autoState = self.autonomousLife.getState() # to know the state of autonomous life
        if autoState != "disabled": # disabled the autonomous life
            self.autonomousLife.setState("disabled")
        self.motionProxy.wakeUp()    # turn on robot's motors
        self.motionProxy.setBreathConfig([["Bpm", 6], ["Amplitude", 0.9]]) # configuration of the breathing
        self.motionProxy.setBreathEnabled("Body", True) # turn on the breathing
        self.motionProxy.setStiffnesses('Head', 1.0) # define the stiffness of the robot's head


        targetName = "Face" # trace humans faces
        faceWidth = 0.1
        self.tracker.registerTarget(targetName, faceWidth)   # register predefined target
        self.tracker.track(targetName)  # start traking process
        self.tts.setVolume(0.8) # adjust the sound
        self.configuration = {"bodyLanguageMode":"contextual"} # start the autonomous life
        #  self.run = True

        global memory
        memory = ALProxy("ALMemory")

    def track(self):
        while self.run:
            time.sleep(1)
            if self.tracker.isTargetLost():
                self.tracker.toggleSearch(True) # search a new target
            else:
                self.tracker.toggleSearch(False) # stop the search of the target

    def setVolume(self, value): # change the volume of Nao robot
        self.tts.setVolume( value )

    def say(self,sentence):
       # self.stopTracker()
        threading.Thread(target = self.animatedSpeechProxy.say, args=(sentence,self.configuration)).start() # define fonctions to execute and start the tread
        self.tracker.track("Face")   # activate the human face track
        self.run = True
     #   self.animatedSpeechProxy.say(sentence,self.configuration)

    def stopTracker(self):
        self.run = False
        self.tracker.stop() # stop to track human face

class vocabulary_TestApp(App):
    def __init__(self):
        super(vocabulary_TestApp, self).__init__()
        self.sound = SoundLoader.load('sounds/9.75.wav')
        self.sound.loop = True

    def playwav(self):
        proc_level = subprocess.Popen([r"C:\anaconda3\envs\python38\python.exe",
                                 r"C:\Users\Administrator\OneDrive - UGent\Desktop\2_load_wav_evalute_other_clips_pann_annoyance/application/main.py"],
                                stdout=subprocess.PIPE, stdin=subprocess.PIPE)

        output_level, error = proc_level.communicate()
        # Define a string representation of a list
        levelstr = str(output_level)
        listlevel = levelstr.strip('[]').split(',')
        print(listlevel)
        self.level = float(listlevel[0].replace("\x1b[0m", ""))
        print('self.level:',self.level)
        # self.level = 2

    def on_keyboard_down(self, window, key, scancode, codepoint, modifier):
        if key == 308:  # Key codes for left and right Ctrl keys
            self.speak_image_button.trigger_action()
            self.txt2.focus = True

        if key == 13:  # Enter key
            self.reward.trigger_action()

        if key == 9:  # Tab key
            if self.txt2.focus:
                self.user_experience.focus = True
                self.txt2.focus = False
            else:
                self.txt2.focus = True
                self.user_experience.focus = False

    def handle_button_release(self, *args):
        pass
        # Do something when the button is released or Enter key is pressed
        # print("Button released or Enter key pressed!")

    def remove_punctuation_and_letters(self, text):
        # Remove punctuation and letters, keep only digits
        text = re.sub(r'[^0-9]', '', text)

        # If no digits found, replace with '3'
        if not text:
            text = '3'

        return text

    def build(self):
        self.playwav()
        self.tts = ALProxy("ALTextToSpeech")  # initialize the robot's speech
        self.tts.setLanguage("English")
        self.motionProxy = ALProxy("ALMotion")  # help make the robot move
        self.tabms = []  # table of ms between two click
        self.tabal = [0]  # table of numbers of values in tabms for one balloon
        self.deb = time.time()
        self.random_word = ''  # if the robot speaks
        self.inputword = ''  # numbers of robot intervention by balloon
        self.p = 0.00  # payment for the participant
        self.nb_letters = 0  # number of correct letters
        self.noise = SoundLoader.load('sounds/1.wav')
        self.c = 0
        self.correct_rate = 0
        self.correct_list = []
        self.user_experience_list = []
        self.nb_bal = 0  # number of used balloon
        self.t = 3  # time between two nao's sentences
        self.files = open(self.ID, "a")  # open the text file
        self.volume_default = 1.27
        self.speed_default = 96
        self.pitchShift_default = 1.49
        self.enunciation_default = 0.46
        self.volumelist = []
        self.speedlist =  []
        self.pitchlist =  []
        self.enunciationlist = []
        # write the text in the file

        self.files.write(
            "number of words, annoylevel, Selected vocabulary,record vocabulary,number of correct letters, world length, correct [0:no/1:yes], payment, volume, speed, pitch, enuaciation, correct rate,user_experience \n")
        game = FloatLayout(padding=10, orientation='vertical')
        game.add_widget(Image(source='images/vocabulary.jpg', allow_stretch=True, keep_ratio=False))

        self.rules = Button(text="rules", size_hint=(None, None), size=(150, 75),
                            pos_hint={'center_x': 0.1, 'center_y': 0.9})
        self.rules.bind(on_press=self.open_rules)
        game.add_widget(self.rules)  # rules button

        self.quit = Button(text='quit', size_hint=(None, None), size=(150, 75),
                           pos_hint={'center_x': 0.9, 'center_y': 0.9})
        self.quit.bind(on_press=self.quits)
        game.add_widget(self.quit)  # quit button

        self.titles = Label(text='[color=#000000]Word recognition game[/color]', size_hint=(None, None), size=(150, 75),
                            pos_hint={'center_x': 0.5, 'center_y': 0.9}, font_size='30sp',markup = True)
        game.add_widget(self.titles)  # title of the game

        self.answer = Label(text="[color=#000000]Please, enter your answer[/color]", font_size='20sp',
                          pos_hint={'center_x': 0.5, 'center_y': 0.35},markup = True)
        game.add_widget(self.answer)  # text on the window

        self.txt2 = TextInput(text='', multiline=False, size_hint=(None, None), size=(300, 35),
                              pos_hint={'center_x': 0.5, 'center_y': 0.3})
        game.add_widget(self.txt2)  # place to write the answer

        self.reward = Button(text='Submit', size_hint=(None, None), size=(100, 50),
                             pos_hint={'center_x': 0.7, 'center_y': 0.3})
        self.reward.bind(on_press=self.change_value)
        self.reward.bind(on_release=self.handle_button_release)
        Window.bind(on_key_down=self.on_keyboard_down)

        game.add_widget(self.reward)  # button to colect the money

        self.nao = Image(source="images/NAO.png", size_hint_x=0.4, size_hint_y=0.4,
                                pos_hint={'center_x': 0.5, 'center_y': 0.55})
        game.add_widget(self.nao)  # balloon deflated / inflated

        self.speak_image = Image(source="images/speak.png", size_hint_x=0.05, size_hint_y=0.05,
                            pos_hint={'center_x': 0.59, 'center_y': 0.68})
        game.add_widget(self.speak_image)

        self.speak_image_button = Button(text='', size_hint=(None, None), size=(50, 50),
                                         background_color =([0, 0, 0, 0]),
                            pos_hint={'center_x': 0.59, 'center_y': 0.68})
        game.add_widget(self.speak_image_button)
        # self.speak_image_button.bind(on_release=self.handle_button_release)
        self.speak_image_button.bind(on_press=self.play)
        Window.bind(on_key_down=self.on_keyboard_down)
        # game.add_widget(self.speak_image_button)

        self.tested = Label(text='Words tested: 0', font_size='25dp', pos_hint={'center_x': 0.9, 'center_y': 0.2},markup = True)
        game.add_widget(self.tested)  # print the number of pump for one balloon

        self.money = Label(text='0.00', font_size='20dp', pos_hint={'center_x': 0.94, 'center_y': 0.4})
        game.add_widget(self.money)  # money win by the participant

        self.pig = Image(source="images/wallet.png", size_hint=(None, None), size=(90, 90),
                         pos_hint={'center_x': 0.93, 'center_y': 0.5})
        game.add_widget(self.pig)  # pig wallet image

        self.livre = Image(source="images/livre.png", size_hint=(None, None), size=(20, 20),
                           pos_hint={'center_x': 0.92, 'center_y': 0.4})
        game.add_widget(self.livre)  # the pound symbol

        self.pleasant = Label(text="[color=#000004]Please rate your pleasantness with the robot voice from 1 to 10[/color]", font_size='20sp',
                            pos_hint={'center_x': 0.5, 'center_y': 0.25}, markup=True)
        game.add_widget(self.pleasant)  # text on the window

        self.user_experience = TextInput(text='', multiline=False, size_hint=(None, None), size=(100, 35),
                              pos_hint={'center_x': 0.5, 'center_y': 0.2})
        game.add_widget(self.user_experience)  # place to write the answer

        return game


    def load_words(self):
        with open('words_alpha.txt') as word_file:
            valid_words = set(word_file.read().split())
            vocabulary_list = list(valid_words)
            # print(type(valid_words))
            # print(type(vocabulary_list))
        return vocabulary_list

    def speak(self, sentence):  # nao speech with no blockage of the rest of the application
        tts = ALProxy("ALTextToSpeech", IP, PORT)  # intializations
        animated = ALProxy("ALAnimatedSpeech", IP, PORT)
        config = {"bodyLanguageMode": "contextual"}
        threading.Thread(target=animated.say, args=(sentence, config)).start()  # nao's speech
        tts.setVolume(self.volume_default)
        tts.setParameter("speed", self.speed_default)
        tts.setParameter("pitchShift", self.pitchShift_default)
        tts.setParameter("doubleVoiceLevel", self.enunciation_default)

    def remove_punctuation_and_digits(self, text):
        # Remove punctuation and digits, keep only letters
        text = re.sub(r'[^a-zA-Z]', '', text)
        return text


    def calculate(self):
        self.txt2.text = self.txt2.text.lower()
        self.txt2.text = self.remove_punctuation_and_digits(self.txt2.text)
        self.random_word = self.random_word.lower()
        if len(self.random_word) >= len(self.txt2.text):
            correct = 0
            for i in range(len(self.txt2.text)):
                if self.txt2.text[i] == self.random_word[i]:
                    correct += 1
        elif len(self.random_word) < len(self.txt2.text):
            correct = 0
            for i in range(len(self.random_word)):
                if self.txt2.text[i] == self.random_word[i]:
                    correct += 1
        return correct

    def play(self, btn):
        english_words = self.load_words()
        # print(type(english_words))
        self.random_word = random.choice(english_words)
        print(self.random_word)
        self.speak(self.random_word)
        # write a function to make nao say the word and speak the single letters
        # for letter in self.random_word:
        #     self.speak(letter)
        self.speak_image.opacity=0
        # self.speak_image_button.text=0

    def change_value(self, btn):  # fonction called by the 'collect reward' button
        self.inputword = self.txt2.text
        if (self.reward.opacity != 0):
            self.speak_image.opacity = 1
            # self.speak_image_button = 1
            self.nb_letters = self.calculate()
            if self.txt2.text != '':
                soundexinputword = phonetics.soundex(self.txt2.text)
                soundexrandomword = phonetics.soundex(self.random_word)
            # Compare the Soundex codes to see if the words have the same pronunciation
                if soundexinputword == soundexrandomword:
                    self.c=1
                    self.correct_rate =1
                    self.nb_letters = len(self.random_word)
                else:
                    if self.nb_letters == len(self.random_word) and self.nb_letters == len(self.inputword):
                        self.c=1
                    else:
                        self.c=0
                        # self.reward_function()
                    self.correct_rate = self.nb_letters/max(len(self.inputword), len(self.random_word))
            else:
                self.c = 0
                self.correct_rate = 0
            self.correct_list.append(self.correct_rate)
            self.user_experience.text = self.remove_punctuation_and_letters(self.user_experience.text)
            self.user_experience_list.append(float(self.user_experience.text))
            print(self.correct_list)
            self.volumelist.append(self.volume_default)
            self.speedlist.append(self.speed_default)
            self.pitchlist.append(self.pitchShift_default)
            self.enunciationlist.append(self.enunciation_default)


            # data = str(self.correct_list)+','\
            #        + str(self.volume_default)+','\
            #        +str(self.speed_default)+','\
            #        +str(self.pitchShift_default)+','\
            #        +str(self.enunciation_default)+','\
            #        +str(self.volumelist) +',' \
            #        + str(self.speedlist) + ',' \
            #        + str(self.pitchlist) +',' \
            #        + str(self.enunciationlist)

            volume_default = 'volume_default,' + str(self.volume_default)
            speed_default = 'speed_default,' + str(self.speed_default)
            pitchShift_default = 'pitchShift_default,' + str(self.pitchShift_default)
            enunciation_default = 'enunciation_default,' + str(self.enunciation_default)

            correct = 'correct_list'
            correct += ''.join([',' + str(each) for each in self.correct_list])

            user_experience = 'user_experience'
            user_experience += ''.join([',' + str(each) for each in self.user_experience_list])

            volume = 'volumelist'
            volume += ''.join([',' + str(each) for each in self.volumelist])

            speed = 'speedlist'
            speed += ''.join([',' + str(each) for each in self.speedlist])

            pitch = 'pitchlist'
            pitch += ''.join([',' + str(each) for each in self.pitchlist])

            enunciation = 'enunciationlist'
            enunciation += ''.join([',' + str(each) for each in self.enunciationlist])

            data = correct + ';' \
                   + user_experience + ';' \
                   + volume_default + ';' \
                   + speed_default + ';' \
                   + pitchShift_default + ';' \
                   + enunciation_default + ';' \
                   + volume + ';' \
                   + speed + ';' \
                   + pitch + ';' \
                   + enunciation

            print('data:',data)  # '[0],0.1,50,1.0,0.25,[0.1],[50],[1.0],[0.25]'
            # subprocess.call([r"C:\anaconda3\envs\python38\python.exe",
            #                  r"C:/Users/Administrator/PycharmProjects/pythonProject1/main.py",data])
            #
            #
            # proc = subprocess.Popen([r"C:\anaconda3\envs\python38\python.exe",
            #                          r"C:/Users/Administrator/PycharmProjects/pythonProject1/ablation.py"],
            #                         stdout=subprocess.PIPE, stdin=subprocess.PIPE)

            # proc = subprocess.Popen([r"C:\anaconda3\envs\python38\python.exe",
            #                          r"C:\Users\Administrator\PycharmProjects\pythonProject1\nnmodel.py"],
            #                         stdout=subprocess.PIPE, stdin=subprocess.PIPE)

            # proc = subprocess.Popen([r"C:\anaconda3\envs\python38\python.exe",
            #                          r"C:\Users\Administrator\OneDrive - UGent\Desktop\qq's document\vacabulary_game\train.py"],
            #                         stdout=subprocess.PIPE, stdin=subprocess.PIPE)

            proc = subprocess.Popen([r"C:\anaconda3\envs\python38\python.exe",
                                     r"C:\Users\Administrator\OneDrive - UGent\Desktop\english-words-master\adam.py"],
                                    stdout=subprocess.PIPE, stdin=subprocess.PIPE)

            proc.stdin.write(data.encode())
            proc.stdin.flush()

            output_data, error = proc.communicate()
            # Define a string representation of a list
            str_list = str(output_data)
            # print("str_list: ", str_list)
            # Use split() method to split the string into substrings
            # based on the delimiter ',' and remove the brackets []
            list_of_strings = str_list.strip('[]').split(',')
            # Convert the list of strings to a list of integers
            my_list = [i for i in list_of_strings]
            my_list[-1] = my_list[-1].replace("]\x1b[0m", "")
            print("my_list: ", my_list)
            # Print the resulting list
            # if self.level > 5:
            #     self.volume_default = float(my_list[-4].split('\r\n[')[-1])
            #     self.volume_default = float(my_list[-4])
            #     self.speed_default = float(my_list[-3])
            #     self.pitchShift_default = float(my_list[-2])
            #     self.enunciation_default = float(my_list[-1])
            #     print('robot:', self.volume_default, self.speed_default, self.pitchShift_default, self.enunciation_default)
            # self.volume_default = float(my_list[-4].split('\r\n[')[-1])
            # self.volume_default = float(my_list[-4])
            # self.speed_default = float(my_list[-3])
            # self.pitchShift_default = float(my_list[-2])
            # self.enunciation_default = float(my_list[-1])
            print('robot:', self.volume_default, self.speed_default, self.pitchShift_default, self.enunciation_default)

            # print(my_list)
            # print('output_data:',type(str(output_data.decode())))
            r = 0
            if (self.nb_bal >= 14):  # if  30 balloons have been used
                tts = ALProxy("ALTextToSpeech", IP, PORT)  # intializations
                tts.setVolume(0.3)
                tts.setParameter("speed", 30)
                tts.setParameter("pitchShift", 0)
                tts.setParameter("doubleVoiceLevel", 0)
                tts.say("this is the end of the game \\pau=500\\ Thank you !")
                # finalvalue = (self.bar.value * 100) / 30
                self.p = self.p + (0.01 * self.nb_letters)  # money earn by participant
                if (len(str(self.p)) < 4):
                    self.money.text = str(self.p) + '0'
                else:
                    self.money.text = str(self.p)
                texts = 'You have finished this game!! \n Thank you for your time \n You earn ' + str(self.p) + ' points'
                self.box4 = FloatLayout(orientation='vertical')
                self.box4.add_widget(Label(text=texts, size_font='20sp', size_hint=(None, None), size=(100, 70),
                                           pos_hint={'center_x': 0.5, 'center_y': 0.75}))
                self.box4.add_widget(Button(text='close the word recognition game', size_hint=(None, None), size=(300, 100),
                                            pos_hint={'center_x': 0.5, 'center_y': 0.25}, on_press=self.close))

                popup = Popup(title='END', title_align='center', title_size='30sp', content=self.box4,
                              size_hint=(0.7, 0.7), auto_dismiss=False)
                popup.open()  # open the end's window
                self.user_experience.text = self.remove_punctuation_and_letters(self.user_experience.text)
                self.files.write("%d , %.2f, %s , %s , %d ,%d , %d , %.2f , %.2f , %.2f,%.2f , %.2f, %.2f,%.2f \n" % (
                    self.nb_bal + 1, self.level, self.random_word, self.inputword, self.nb_letters, len(self.random_word),self.c, self.p,
                    self.volume_default, self.speed_default, self.pitchShift_default, self.enunciation_default, self.correct_rate, float(self.user_experience.text)))
                deb2 = time.time()  # take the time
                self.deb = (deb2 - self.deb) * 1000  # pass in millisecond
                self.tabms += [int(self.deb)]
                self.deb = deb2
                l = len(self.tabms)
                tps2 = time.time()
                timer = round(tps2 - self.tps1, 2)
                with open(self.ID, "r+") as file:  # complete the file
                    text = file.read()
                    i = text.index("XXXX")  # end time
                    j = text.index("ZZZZ")  # time pass on the game
                    k = text.index("PP.PP")  # money win by the participant
                    file.seek(0)  # comme back at the start of the file
                    file.write(
                        text[:i] + str(time.strftime('%H:%M', time.localtime())) + text[i + 4: j] + str(timer) + text[
                                                                                                                 j + 5:k] + str(
                            self.p) + text[k + 5:])

            else:  # if it's not the end of the game
                self.inputword = self.txt2.text
                self.speak_image.opacity = 1
                # self.speak_image_button = 1
                deb2 = time.time()
                self.deb = (deb2 - self.deb) * 1000  # pass to second in milisecond
                self.tabms += [int(self.deb)]  # add the value in the tab
                self.deb = deb2
                l = len(self.tabms)
                self.tabal += [l]
                self.p = self.p + (0.01 * int(self.nb_letters))  # reward for the player
                if (len(str(self.p)) < 4):
                    self.money.text = str(self.p) + '0'
                else:
                    self.money.text = str(self.p)
                # self.files.write("%d , %s , %s , %d , %d, %d , %.2f \n" % (
                # self.nb_bal + 1, self.random_word, self.inputword, self.nb_letters,len(self.random_word),self.c, self.p))
                self.user_experience.text = self.remove_punctuation_and_letters(self.user_experience.text)
                self.files.write("%d , %.2f, %s , %s , %d ,%d , %d , %.2f , %.2f , %.2f,%.2f , %.2f, %.2f, %.2f \n" % (
                self.nb_bal + 1, self.level, self.random_word, self.inputword, self.nb_letters, len(self.random_word), self.c, self.p,
                self.volume_default, self.speed_default, self.pitchShift_default, self.enunciation_default,self.correct_rate, float(self.user_experience.text)))
                self.nb_bal += 1
                self.tested.text = 'Word tested:' + str(self.nb_bal)
                self.nb_letters = 0
                self.txt2.text = ''
                self.user_experience.text = ''

    def talk(self, v):  # Nao movements and talking after an explosion
        JointNames = ["RShoulderPitch", "RShoulderRoll", "LShoulderPitch",
                      "LShoulderRoll"]  # initialisation for right arm and left arm
        Arm1 = [60, -20, 60, 20]  # position of right and left arms
        Arm1 = [x * motion.TO_RAD for x in Arm1]
        pFractionMaxSpeed = 0.5  # speed of movement
        self.motionProxy.angleInterpolationWithSpeed(JointNames, Arm1, pFractionMaxSpeed)  # movement
        self.speak("ho !")
        Arm1 = [80, 5, 80, 5]
        Arm1 = [x * motion.TO_RAD for x in Arm1]
        pFractionMaxSpeed = 0.3
        self.motionProxy.angleInterpolationWithSpeed(JointNames, Arm1, pFractionMaxSpeed)

    def pop(self, btn):  # open the end popup
        self.popup.open()

    def open_rules(self, btn):  # fonction called by the 'rules' button
        popup = Popup(title='rules', title_align='center', title_size='30sp', content=Label(text='''
            we are going to have a word recognition game.one after the other.
            The robot will randomly select a word and pronounce it at a time.
            You will have a limited amount of time to type in your answer.
            You can only submit one answer per word.
            Each correct letter earns you a point,
            and incorrect or incomplete answers receive no points
            You can click "Submit" button to submit your answer upon you finish your answer.
            and start with a new word. Remember, you have 35 words in total.
            Attention: if you quit the test before the end, you will not earn anything.
            If you have questions, it is the right time to ask! Good luck! '''), size_hint=(None, None),
                      size=(650, 450))  # create the popup window
        popup.open()  # open the rules window

    def quits(self, btn):  # fonction called by the 'quit' button, asks a confirmation before quit the game
        self.tts = ALProxy("ALTextToSpeech")
        self.box = FloatLayout(orientation='vertical')
        testq = '''
        Are you sure you want to quit ?
        You will not receive any money'''
        self.speak("Are you sure you want to quit me ?")  # nao speaks
        self.box.add_widget(Label(text=testq, size_font='20sp', size_hint=(None, None), size=(100, 70),
                                  pos_hint={'center_x': 0.5, 'center_y': 0.75}))
        self.box.add_widget(
            Button(text='yes', size_hint=(None, None), size=(120, 100), pos_hint={'center_x': 0.25, 'center_y': 0.3},
                   on_press=self.yes))
        self.box.add_widget(
            Button(text='no', size_hint=(None, None), size=(120, 100), pos_hint={'center_x': 0.75, 'center_y': 0.3},
                   on_press=self.no))
        self.popup = Popup(title='Exit', content=self.box, size_hint=(None, None), size=(400, 300), auto_dismiss=False)
        self.popup.open()  # open the popup

    def no(self, btn):  # if the participant comes back on the game
        self.popup.dismiss()
        self.speak("cool !")

    def yes(self, btn):  # if the participant wants quit the game before the end
        self.speak("ok, good bye")
        texts = 'you have finished this game,thank you for your time \n + % \n '
        self.box2 = FloatLayout(orientation='vertical')
        self.box2.add_widget(Label(text=texts, size_font='20sp', size_hint=(None, None), size=(100, 70),
                                   pos_hint={'center_x': 0.5, 'center_y': 0.75}))
        self.box2.add_widget(Button(text='close the word recognition game', size_hint=(None, None), size=(200, 100),
                                    pos_hint={'center_x': 0.5, 'center_y': 0.25}, on_press=self.close))
        popup = Popup(title='END', content=self.box2, size_hint=(0.7, 0.7), auto_dismiss=False)
        popup.open()  # open the popup window
        self.user_experience.text = self.remove_punctuation_and_letters(self.user_experience.text)
        self.files.write("%d , %.2f, %s , %s , %d ,%d , %d , %.2f , %.2f , %.2f,%.2f , %.2f, %.2f,  %.2f \n" % (
            self.nb_bal + 1, self.level, self.random_word, self.inputword, self.nb_letters, len(self.random_word), self.c, self.p,
            self.volume_default, self.speed_default, self.pitchShift_default, self.enunciation_default,
            self.correct_rate,float(self.user_experience.text)))
        self.files.write("quit button used")
        self.files.close()  # close the file
        tps2 = time.time()
        timer = round(tps2 - self.tps1, 2)  # compt seconds passed on the test
        with open(self.ID, "r+") as file:  # 'r+' to read and write on the file
            text = file.read()  # read the file
            i = text.index("XXXX")  # note the starting of the "XXXX" chain
            j = text.index("ZZZZ")
            k = text.index("PP.PP")
            file.seek(0)  # comme back at the start of the file
            file.write(text[:i] + str(time.strftime('%H:%M', time.localtime())) + text[i + 4: j] + str(timer) + text[
                                                                                                                j + 5:k] + str(
                00.00) + text[k + 5:])
        self.motionProxy.rest()  # Nao turns off its motors

    def close(self, btn):
        App.get_running_app().stop()
        self.rules.opacity = 0
        self.quit.opacity = 0
        self.titles.opacity = 0
        self.reward.opacity = 0
        self.sound.stop()
        self.sound.unload()
        App.get_running_app().stop()

class StartApp(App):
    def build(self):

        layout = FloatLayout(padding=10, orientation='vertical')
        layout.add_widget(Image(source='images/vocabulary.jpg', allow_stretch=True, keep_ratio=False))

        self.btn1 = Button(text="start", size_hint=(None, None), size=(200, 100),
                           pos_hint={'center_x': 0.5, 'center_y': 0.25})
        self.btn1.bind(on_press=self.buttonClicked)
        layout.add_widget(self.btn1)  # 'start' button

        self.lbl1 = Label(text="[color=#000000]Please, enter your participant ID[/color]", font_size='30sp',markup = True,
                          pos_hint={'center_x': 0.5, 'center_y': 0.75})
        layout.add_widget(self.lbl1)  # text on the window

        self.txt1 = TextInput(text='', multiline=False, size_hint=(None, None), size=(300, 35),
                              pos_hint={'center_x': 0.5, 'center_y': 0.5})
        layout.add_widget(self.txt1)  # place to write the participant ID


        return layout

    def opac(self, x):
        x = 1
        self.btn1.opacity = 1
        # self.English.opacity=1
        # self.Dutch.opacity=1

    def part_2(self, x):  # differents paragraphs of rules
        self.lbl1.text = '''
        [color=#000000]The robot will randomly select a word
        and pronounce it at a time.[/color] '''

    def part_3(self, x):
        self.lbl1.text = '''
        [color=#000000]You need to type in your answer, 
        and your pleasantness about the robot voice from 1 to 10[/color]. '''

    def part_4(self, x):
        self.lbl1.text = '''
        [color=#000000]You can only submit one answer per word.
        Each correct letter earns you a point,
        and incorrect or incomplete answers receive no points[/color]. '''

    def part_5(self, x):
        self.lbl1.text = '''
        [color=#000000]You can click "Submit" button to submit your answer
        upon you finish it.
        and start with a new word.
        Remember, you have 35 words in total.[/color]  '''

    def part_6(self, x):
        self.lbl1.text = '''
        [color=#000000]Attention:
        if you quit the test before the end, you will not earn anything.[/color]'''

    def part_7(self, x):
        self.lbl1.text = '''
        [color=#000000]If you have questions, it is the right time to ask! Good luck![/color] '''


    def buttonClicked(self, btn):

        if (self.btn1.text == "start"):
            bla = self.txt1.text
            fileID = path + str(bla) + ".txt"  # create file's name with the participant ID
            if (exists(fileID)):
                self.lbl1.text = "[color=#000000]This ID already exist. Please choose an other ID[/color]"
            else:
                files = open(fileID, "a")  # open the text file
                files.write("Participant ID : %s \nNoise level :  \n" % (
                    bla))  # write in the text file
                times = "Date : " + str(time.strftime('%d/%m/%y', time.localtime())) + "\nStart time : " + str(
                    time.strftime('%H:%M', time.localtime())) + "\n"
                files.write(times)
                vocabulary_TestApp.tps1 = time.time()
                files.write("End time : XXXX \nElapsed time : [s] ZZZZ \nTotal reward : PP.PP \n")
                files.close()  # close the text files
                self.lbl1.font_size = '25sp'
                self.lbl1.pos_hint = {'center_x': 0.5, 'center_y': 0.65}
                self.lbl1.text = '''
                [color=#000000]We are going to have a word recognition game. one after the other.[/color] '''  # change the text on the window
                try:
                    self.myBroker = ALBroker("myBroker", "0.0.0.0", 0, IP, PORT)  # robot part
                except RuntimeError:
                    self.myBroker = ALBroker("myBroker", "0.0.0.0", 0, IP, PORT)
                self.module = RobotControllerModule("module")
                # self.postureProxy = ALProxy("ALRobotPosture")

                self.trackerThread = threading.Thread(target=self.module.track)
                self.trackerThread.start()  # activate a thread
                tts = ALProxy("ALTextToSpeech", IP, PORT)  # intializations
                tts.setVolume(0.3)
                tts.setParameter("speed", 20)
                tts.setParameter("pitchShift", 1)
                tts.setParameter("doubleVoiceLevel", 0)
                # # Changes the basic voice of the synthesis
                #
                # lang = tts.getAvailableVoices();
                # print "Available voice: " + str(lang)
                # tts.setVoice("Kenny22Enhanced")
                # self.module.say('''\\rspd=75\\ we are going to have a vocabulary test.\\pau=200\\one after the other.  ''')
                self.module.say('''\\rspd=75\\ We are going to have a word recognition game.\\pau=300\\
                                  one after the other.\\pau=500\\
                                 The robot will randomly select a word and pronounce it at a time.\\pau=1000\\
                                You need to type in your answer, 
                                and your pleasantness about the robot voice from 1 to 10. \\pau=500\\
                                You can only submit one answer per word. \\pau=500\\
                                Each correct letter earns you a point,\\pau=500\\
                                 and incorrect or incomplete answers receive no points\\pau=500\\
                                You can click "Submit" button to submit your answer upon you finish it.\\pau=800\\
                                and start with a new word. Remember,\\pau=500\\
                                 you have 35 words in total. \\pau=500\\
                                Attention:\\pau=500\\ if you quit the test before the end, you will not earn anything.\\pau=2000\\
                                If you have questions, it is the right time to ask! Good luck!  ''')


                # presentation of the game by nao \\pau\\ make pauses during its speech, \\rspd\\ is for the output of the Nao's speech

                self.btn1.opacity = 0  # hidden the 'start' button
                self.btn1.text = "start the test"  # turn 'start' into 'start game'
                self.btn1.pos_hint = {'center_x': 0.5, 'center_y': 0.15}
                self.btn1.size = (150, 150)  # change the size of the button
                self.txt1.size = (0, 0)  # reduce the TextInput
                self.txt1.opacity = 0  # hidden the TextInput
                Clock.schedule_once(partial(self.part_2), 8)  # post rules paragraph per paragraph
                Clock.schedule_once(partial(self.part_3), 14)
                Clock.schedule_once(partial(self.part_4), 23)
                Clock.schedule_once(partial(self.part_5), 36)
                Clock.schedule_once(partial(self.part_6), 47)
                Clock.schedule_once(partial(self.part_7), 56)
                Clock.schedule_once(partial(self.opac), 61)  # reapper the 'start game' butto

        elif (self.btn1.text == "start the test"):
            App.get_running_app().stop()  # close the starting window
            vocabulary_TestApp.ID = path + str(self.txt1.text) + ".txt"  # to change on an other computer
            vocabulary_TestApp().run()  # call the game

Window.fullscreen = False
StartApp().run()




