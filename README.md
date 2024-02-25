# robot-speech-intelligibility
This repository contains the dataset and code associated with the paper "Cool glasses, where did you get them?": Generating Visually Grounded conversation starters for Human-Robot Dialogue, by Qiaoqiao Ren, Yuanbo Hou, Thomas Demeester, and Tony Belpaeme.

In this work, we built a adaptive system that generates questions for robots to start an open-domain conversation with a user, based on visual information.

We collected a data set of 4000 images that are appropriate for Human-Robot Interaction (HRI), meaning they are similar to what a robot camera would see in an interaction with a user. Each image is accompanied by three conversation-starting questions, that refer to something in the image.

Then, we compared two systems to generate these questions: a retrieval-based model as baseline, and the Transformer-based encoder-decoder model BART, fine-tuned on our data set. Both models were trained and evaluated using ParlAI. As input for the question-generating models, we used a dense captioning model to generate a description of the image.

Finally, we deployed the system on a Furhat social robot.

This README contains technical instructions to replicate our results. Read the Nao text to speech official document for more information, or the paper itself if you want details about the methodology! For any further questions, do not hesitate to contact Qiaoqiao[dot]Ren[at]ugent[dot]be.

