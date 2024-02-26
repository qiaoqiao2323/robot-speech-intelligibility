# robot-speech-intelligibility
This repository contains the dataset and code associated with the paper "No More Mumbles: Enhancing Robot Intelligibility through Speech Adaptation", by Qiaoqiao Ren, Yuanbo Hou, Thomas Demeester, and Tony Belpaeme.

In this work, we built an adaptive system that improves the robot's speech intelligibility in various spaces to adapt to different users.

Firstly, we build an annoyance level prediction model in t

To reveal how ambient sound, the environment’s acoustic quality, the distance between the user and the robot, and the user’s hearing can affect both the robot’s intelligibility and the user’s experience, we first set up a data collection campaign.

Then, we compared two systems to generate these questions: a retrieval-based model as baseline, and the Transformer-based encoder-decoder model BART, fine-tuned on our data set. Both models were trained and evaluated using ParlAI. As input for the question-generating models, we used a dense captioning model to generate a description of the image.

Finally, we deployed the system on a Nao social robot.

This README contains technical instructions to replicate our results. For any further questions, do not hesitate to contact Qiaoqiao[dot]Ren[at]ugent[dot]be.

