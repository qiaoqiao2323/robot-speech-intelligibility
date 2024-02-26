# Visual Conversation Starters

This repository contains the dataset and code associated with the paper *"Cool glasses, where did you get them?": Generating Visually Grounded conversation starters for Human-Robot Dialogue*, by Ruben Janssens, Pieter Wolfert, Thomas Demeester, and Tony Belpaeme, [published at HRI'22](https://dl.acm.org/doi/abs/10.5555/3523760.3523884).

In this work, we built a system that generates questions for robots to start an open-domain conversation with a user, based on visual information.

We collected a data set of 4000 images that are appropriate for Human-Robot Interaction (HRI), meaning they are similar to what a robot camera would see in an interaction with a user. Each image is accompanied by three conversation-starting questions, that refer to something in the image.

Then, we compared two systems to generate these questions: a retrieval-based model as baseline, and the Transformer-based encoder-decoder model BART, fine-tuned on our data set. Both models were trained and evaluated using ParlAI. As input for the question-generating models, we used a dense captioning model to generate a description of the image.

Finally, we deployed the system on a Furhat social robot.

This README contains technical instructions to replicate our results. Read our [blog post](https://rubenjanssens.be/visual-conversation-starters) for more information, or the [paper](https://dl.acm.org/doi/abs/10.5555/3523760.3523884) itself if you want details about the methodology! For any further questions, do not hesitate to contact `rmajanss[dot]janssens[at]ugent[dot]be` or [@rubenjanss](https://www.twitter.com/rubenjanss)  on Twitter.

If you use our work, please cite our paper using the following citation:
```
@inproceedings{janssens2022cool,
  title={“Cool glasses, where did you get them?” Generating Visually Grounded Conversation Starters for Human-Robot Dialogue},
  author={Janssens, Ruben and Wolfert, Pieter and Demeester, Thomas and Belpaeme, Tony},
  booktitle={Proceedings of the 2022 ACM/IEEE International Conference on Human-Robot Interaction},
  pages={821--825},
  year={2022}
}
```

## Data set

`data/` contains the full data set. The images are selected from the [YFCC100M data set](http://www.multimediacommons.org/) and are not hosted in this repository, rather, we refer to the images using their Flickr URL.

The data set is split into a training set (3k images), validation set (500 images), and test set (500 images). Each subset is represented as a JSON file with the following members, each with one entry per image:

- **Answer.question1**: first question associated with the image
- **Answer.question2**: second question associated with the image
- **Answer.question3**: third question associated with the image
- **url**: Flickr URL of the image
- **name**: filename of the image (substring of the URL)
- **dot_string**: description of the image, generated by the dense captioning model.

We have also included a modified version of the training set, called `
train_dataset_lowfreq.json `. This data set contains all images and questions of the original, *except* for the six questions that were most common in the training set, each occurring more than 50 times. We found that the system generated more diverse questions after being trained on this reduced data set.

## Captioning

The image descriptions were generated by the *Dense Captioning with Joint Inference and Visual Context* model, from [https://github.com/linjieyangsc/densecap](https://github.com/linjieyangsc/densecap). We used their official sample model and did not fine-tune it further. The model ran on a machine with an NVIDIA GeForce GTX 1080 Ti GPU with 11GB VRAM.

`captioning/` contains code that we used for running the captioning model.

- `test.py` is a modified version of `lib/fast_rcnn/test.py` in the densecap repo: replace this file with our file.
- `batch.py` is a modified version of `lib/tools/demo.py` for generating captions for a batch of images.

- `caption_server.py` is a modified version of `lib/tools/demo.py` for live caption generation, e.g. when running the system with Furhat.

To run our code, first clone the original repository and follow their install instructions. Replace the `test.py`file as described above. Then, run `batch.py` with the following arguments: `python batch.py --image_folder <IMAGE_FOLDER> --output <OUTPUT_FILE> --image_file <DATASET_FILE>`. `DATASET_FILE` is a JSON file with a data set containing the `url` and `name` fields as in our data set. The script will download all images into `IMAGE_FOLDER`.

## Training and Evaluating using ParlAI

`parlai_internal/` contains the code needed to train and evaluate on our task in the ParlAI framework. Check the original [ParlAI repository](https://github.com/facebookresearch/ParlAI) for more information on how to use the ParlAI framework and how to use a `parlai_internal` folder to define custom tasks.

The folder contains two tasks: `text_opener` and `text_opener_lowfreq`. They present the captions as input for the ParlAI model and the conversation-starting question as expected output. `text_opener_lowfreq` uses the training set without the six most common questions, as described above. `text_opener` uses the full data set.

The code for these tasks is based on the `mnist_qa` task included in ParlAI.

Place the data set files in a folder `data/opener_text/` in your ParlAI folder.

We used the following commands to train and evaluate the BART model on this task:

```
parlai train_model -m bart --init-model zoo:bart/bart_large/model -mf <MODEL_OUTPUT_FILE> -t internal:opener_text_lowfreq -bs 24 --fp16 true -eps 10 -lr 1e-6 --optimizer adam --inference beam --beam-size 5 --validation-every-n-epochs 8 --metrics all --validation-metric bleu-4
```
```
parlai eval_model -mf <TRAINED_MODEL_FILE> -t internal:opener_text -dt test -rf <EVALUATION_OUTPUT_FILE> --save-world-logs True --inference beam --beam-size 5
```

We trained the BART model using an NVIDIA Tesla V100 GPU, with 32GB of VRAM. Inference was done using an NVIDIA GeForce GTX 1080 Ti GPU with 11GB VRAM.

The baseline retrieval model was trained and evaluated using these commands (using only a CPU):

```
parlai train_model -m tfidf_retriever -t internal:opener_text_lowfreq -mf <MODEL_OUTPUT_FILE> -dt train:ordered -eps 1 --retriever-tokenizer simple --retriever-ngram 3
```
```
parlai eval_model -t internal:opener_text -mf <TRAINED_MODEL_FILE> -dt test --metrics all -rf <EVALUATION_OUTPUT_FILE> --save-world-logs True
```

Model weights can be made available upon request.

## Running on Furhat

Finally, you can also run a live demo of the system on a physical (or virtual) Furhat robot. For this, you need to run the following code. The python scripts require the `pyzmq` package.

- `VisualConversationStarter/` contains the skill code that should run on the Furhat. You can find instructions on how to run a skill on the robot in the [Furhat documentation](https://docs.furhat.io/skills/#running-a-skill-on-a-robot). Inspiration for the code was found [in this example skill](https://github.com/FurhatRobotics/tutorials/tree/main/camerafeed-demo).
- `captioning/captioning_server.py` should be run to generate the captions. No command-line arguments are needed. Follow the instructions above to set up the captioning model.
- For the question-generating model, run the following command in an environment where you have set up ParlAI and our task(s) as described above. Edit `parlai_internal/config.yml` so `model_file` contains the correct path to your trained model.
```
python3 ~/ParlAI/parlai/chat_service/services/browser_chat/run.py --config-path ~/ParlAI/parlai_internal/config.yml --port <QUESTION_GENERATION_PORT>
```

- `demo/visual_conversation_starter_server.py` is the script that integrates all components: it passes the data around between the Furhat and the models. Before running the script, make sure the IP addresses and ports for all components are correct in the `config` dictionary.

## Contact

For any further questions, do not hesitate to contact rmajanss[dot]janssens[at]ugent[dot]be or [@rubenjanss](https://www.twitter.com/rubenjanss) on Twitter. You can also always raise an issue in the repository.

<!--<script> window.scroll(0,100000) </script> -->
