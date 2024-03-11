# Robot-speech-intelligibility

This repository contains the dataset and code associated with the paper "No More Mumbles: Enhancing Robot Intelligibility through Speech Adaptation", by Qiaoqiao Ren, Yuanbo Hou, Thomas Demeester, and Tony Belpaeme.

In this work, we built an adaptive system that improves the robot's speech intelligibility in various spaces to adapt to different users.

Firstly, we build an annoyance level prediction model based on the ambient noise.

To reveal how ambient sound, the environment’s acoustic quality, the distance between the user and the robot, and the user’s hearing can affect both the robot’s intelligibility and the user’s experience, we first set up a data collection campaign.

Finally, we deployed the system on a Nao social robot.

This README contains technical instructions to replicate our results. For any further questions, do not hesitate to contact Qiaoqiao[dot]Ren[at]ugent[dot]be.


## Ambient sounds’ annoyance rating prediction

`Annoyance_rating_prediction/` contains the training, validation, and testing dataset and the pretrained model for the ambient sounds’ annoyance rating prediction.

To train the ARP model to successfully infer the impact of environmental sounds on participants, i.e., the annoyance level for participants, based on acoustic features containing amplitude, frequency, and category information of ambient sounds, we used a real-life polyphonic audio dataset, [DeLTA (Deep Learning Techniques for noise Annoyance detection) Dataset](https://zenodo.org/records/7158057) which is not hosted in this repository, rather, we refer to the data using their URL. The DeLTA contains 2980 samples of 15-second binaural audio recordings with 24 classes of sound sources from European cities for noise annoyance prediction. A remote listening experiment was conducted, which involved participants listening to 15-second binaural recordings of urban environments and being instructed to identify the sound sources present within each recording and to provide an annoyance rating on a scale of 1 to 10. 

The proposed annoyance rating prediction (ARP) is trained with the GPU (Tesla V100-SXM2-32GB) and the CPU (Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz). In inference, taking the GPU as Tesla V100-SXM2-32GB and the CPU as Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz as an example, the response time of the model from inputting acoustic features to outputting prediction results is 1.87 ms, which means the model can process input sounds in real-time.

To run the pretrained ARP models, please refer to `Annoyance_rating_prediction/.`

## Word recognition game

`word_recognition/` contains code that we used for running the captioning model.
`NU_6_words/` contains open-set recognition of four lists of English consonant-nucleus-consonant (CNC) monosyllabic words (e.g. hash, dodge, should) from the Northwestern University Auditory Test 6 (NU-6)
 

- `main.py` is the word recognition task main program for data collection.
- `game_nointro.py` is a modified version of `word_recognition/main.py` to skip the introduction spoken by the robot.


##################################################### Installation ################################################

To install : Python 2.7 (https://www.python.org/downloads/)
	     kivy library (https://kivy.org/#download)
	     naoqi library


Open the file "data.txt" if you have chosen to use a program with a robot.
Turn on the robot with a press for a few seconds on the button located on its torso.
Press for a short time on this button when the robot is turned on and the robot is going to tell you its IP address.
Modify the IP address in the file by replacing it with the IP address given by the robot.
The IP address should be like : 192.168.1.13
Save the "data.txt" file.

To run our code, first clone the original repository and follow their install instructions. Replace the `main.py`file as described below.
We used the following commands to train and evaluate the BART model on this task:

```
python main.py
```

A window will appear on the screen requesting the ID of the participant.
The participant ID permits to save different observations collected during the word recognition test in a file text proper to 
the participant.
Two participants can't have the same ID.

To collect each information, close the window by just using the button "close the BART test".
The button "close the test" appears in the end window when the test is over.
To show the end window, you can use two methods :
	1.Finish the test after using 50 words.
	2.Use the "quit" button and after push the "yes" button to confirm the end of the experiment.

If another  method is used to close the test, some data will be lost (points earned, time in millisecond between
two clicks, the time in seconds on the game and the time of the end of the experiment).

To access at results files, go to the "word_recognition" folder and after in the "results" folder.
Names of results files are in the shape of "results_participantID.txt".

#################### Rules the test #########################################

Rules of the game:
We are going to have a word recognition game. one after the other. The robot will randomly select a word and pronounce each letter one at a time.
You need to type in your answer, and your pleasantness about the robot voice from 1 to 10. You can only submit one answer per word.
Each correct letter earns you a point, and incorrect or incomplete answers receive no points, you can click "Submit" button to submit your answer upon you finish it.
and start with a new word. Remember, you have 50 words in total.
Attention: if you quit the test before the end, you will not earn anything. If you have questions, it is the right time to ask! Good luck! 

## Pretained model

`Pretrained_model/` contains the source code and pre-trained adaptive speech model



## Evaluation

To bridge the communication between Python 2.7 (NaoQi) and Python 3.8 (ETV and JMRE model), the subprocess module facilitates communication. 

- `Evaluation.py` is the word recognition task for evaluation process.


We used the following code in the script to take the annoyance rating from the APR model as input of the adaptive speech model:

```
            proc = subprocess.Popen([r"C:\anaconda3\envs\python38\python.exe",
                                     r"C:\Users\Administrator\OneDrive - UGent\Desktop\english-words-master\adam.py"],
                                    stdout=subprocess.PIPE, stdin=subprocess.PIPE)
```

We trained the adaptive speech model using a CPU.

## Demos

`Demos/` contains the demo of adaptive speech condition, robot default voice, and the recommend threshold of the slower speech speed.

## Running on Nao

Finally, you can also run a live demo of the system on a physical (or virtual) Furhat robot. For this, you need to run the following code. The python scripts require the `pyzmq` package.

You can find instructions on how to run a skill on the robot in the [Furhat documentation](https://docs.furhat.io/skills/#running-a-skill-on-a-robot). 
- `captioning/captioning_server.py` should be run to generate the captions. No command-line arguments are needed. Follow the instructions above to set up the captioning model.
- For the question-generating model, run the following command in an environment where you have set up ParlAI and our task(s) as described above. Edit `parlai_internal/config.yml` so `model_file` contains the correct path to your trained model.
```
python3 ~/ParlAI/parlai/chat_service/services/browser_chat/run.py --config-path ~/ParlAI/parlai_internal/config.yml --port <QUESTION_GENERATION_PORT>
```

- `demo/comparison.txt` that integrates all components: it passes the data around between the Furhat and the models. Before running the script, make sure the IP addresses and ports for all components are correct.

## Contact

For any further questions, do not hesitate to contact Qiaoqiao[dot]Ren[at]ugent[dot]be or Yuanbo[dot]Hou[at]ugent[dot]be. You can also always raise an issue in the repository.


<!--<script> window.scroll(0,100000) </script> -->
