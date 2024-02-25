import torch
import sys
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import math


import sys


def str2float(correct_list):
    return [float(each) for each in correct_list]

# Read input from stdin
input_data = sys.stdin.readline()

part_input_data = input_data.split(';')
correct_list = part_input_data[0].split('correct_list,')[1].split(',')
# print(correct_list)
feedback_list = str2float(correct_list)


user_list = part_input_data[1].split('user_experience,')[1].split(',')
user_list = str2float(correct_list)

volume_default = part_input_data[2].split('volume_default,')[1]
speed_default = part_input_data[3].split('speed_default,')[1]
pitchShift_default = part_input_data[4].split('pitchShift_default,')[1]
enunciation_default = part_input_data[5].split('enunciation_default,')[1]

Volume, speed, pitch, enunciation = float(volume_default), float(speed_default), float(pitchShift_default), float(enunciation_default)

volumelist = part_input_data[6].split('volumelist,')[1].split(',')
volumelist = str2float(volumelist)
# print(volumelist)

speedlist = part_input_data[7].split('speedlist,')[1].split(',')
speedlist = str2float(speedlist)
# print(speedlist)

pitchlist = part_input_data[8].split('pitchlist,')[1].split(',')
pitchlist = str2float(pitchlist)
# print(pitchlist)

enunciationlist = part_input_data[9].split('enunciationlist,')[1].split(',')
enunciationlist = str2float(enunciationlist)
# print(enunciationlist)


feedback = feedback_list[-1]
# print('feedback',feedback)
user = user_list[-1]


VOLUME_RANGE = [0.1, 1]
SPEED_RANGE = [50, 400]
PITCH_RANGE = [1, 4]
ENUNCIATION_RANGE = [0.01, 0.5]

# Define the features set to the robot
last_parameters = [Volume, speed, pitch, enunciation]
# print('last_parameters: ', last_parameters)

# Define the reward and cost functions as PyTorch modules
# class RewardFunction(nn.Module):         #MAE
#     def forward(self, y_true):
#         if len(feedback_list) >=2:
#             if y_true == 0:
#                 episo = 1e-4
#                 feeback_difference = y_true - float(feedback_list[-2])
#                 if feeback_difference <= 0:
#                     reward = 1/episo
#                     reward = torch.tensor(float(reward))
#                 else:
#                     reward = 1 / (episo + feeback_difference)
#                     reward = torch.tensor(float(reward))
#             else:
#                 episo = 1e-2
#                 feeback_difference = y_true - float(feedback_list[-2])
#                 if feeback_difference <= 0:
#                     reward = 1/episo
#                     reward = torch.tensor(float(reward))
#                 else:
#                     reward = 1/(episo+feeback_difference)
#                     reward = torch.tensor(float(reward))
#         else:
#             if y_true == 0:
#                 episo = 1e-4
#                 reward = 1 / (episo + y_true)
#                 reward = torch.tensor(float(reward))
#             else:
#                 episo = 1e-2
#                 reward = 1 / (episo + y_true)
#                 reward = torch.tensor(float(reward))
#         return reward

class RewardFunction(nn.Module):
    def forward(self):
        reward = torch.tensor(10-float(feedback) * float(user))
        return reward


def memorybuffer():
    feedback_memory = feedback_list
    user_memory = user_list
    action_memory = volumelist,speedlist,enunciationlist,pitchlist
    return feedback_memory,action_memory,user_memory


def Trustregion(parameterlist, RANGE):
    rlist=[]
    index_increcement = 1
    for each in range(len(feedback_list)):
        if feedback_list[each] == 1:
            if user_list[each] != 10:
                rlist.append(parameterlist[each])
                rlistsort = list(set(rlist))
                rlistsort.sort()
                middle_index = len(rlistsort) // 2
                if len(rlist) > 1:
                    RANGE[0] = rlistsort[middle_index -1]
                    RANGE[1] = rlistsort[middle_index]
                else:
                    RANGE[0] =  RANGE[0]
                    RANGE[1] =  RANGE[1]
                index_increcement = ((len(rlist) // 3) + 1)
            else:
                index_increcement = 100
    return RANGE[0] , RANGE[1], index_increcement

# Initialize the machine learning nb model as a PyTorch module
class PolicyModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PolicyModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.b1 = nn.Linear(hidden_size, 2)

        self.b2 = nn.Linear(hidden_size, 2)

        self.b3 = nn.Linear(hidden_size, 2)

        self.b4 = nn.Linear(hidden_size, 2)

        self.b5 = nn.Linear(hidden_size, 1)

        self.b6 = nn.Linear(hidden_size, 1)


    def forward(self, state):
        self.state = torch.relu(self.linear1(state))
        self.common = torch.relu(self.linear2(self.state))

        b1 = F.softmax(self.b1(self.common), dim=1)
        b2 = F.softmax(self.b2(self.common), dim=1)
        b3 = F.softmax(self.b3(self.common), dim=1)
        b4 = F.softmax(self.b4(self.common), dim=1)
        b5 = 1 * torch.sigmoid(self.b5(self.common))
        b6 = 10.0 * torch.sigmoid(self.b6(self.common))
        result = torch.cat([b1, b2, b3, b4, b5, b6], dim=-1)

        # print('result:', result.size())

        return result

model = PolicyModel(input_size=6, hidden_size=20)
criterion_reward = RewardFunction()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def select_action(y_pred_list):
    # print('state:',state)
    # y_pred_list = model(state).squeeze().tolist()
    # print('y_pred', y_pred)
    # y_pred = [-1 if each < 0.5 else 1 for each in y_pred]

    # action = [1, -1]
    action = []
    probality = []
    for i in range(0, len(y_pred_list), 2):
        if y_pred_list[i] > y_pred_list[i + 1]:
            action.append(y_pred_list[i])
            probality.append((y_pred_list[i]))
        else:
            action.append(-y_pred_list[1 + i])
            probality.append(y_pred_list[i + 1])
            probality = [each for each in probality]
    return action, probality

# Make a prediction with the current model and the last state of parameters
# state = torch.tensor(last_parameters + [feedback] +[user], dtype=torch.float32).unsqueeze(0)
action= []
probality = []
# action,probality = select_action(state)

# volumelist = torch.tensor(volumelist)
# speedlist = torch.tensor(speedlist)
# pitchlist = torch.tensor(pitchlist)
# enunciationlist = torch.tensor(enunciationlist)
# targets = torch.tensor(feedback_list)
# user_list = torch.tensor(user_list)

# Combine your inputs into a single tensor
# stack the input lists into a tensor
inputs = torch.stack([torch.tensor(volumelist),
                      torch.tensor(speedlist),
                      torch.tensor(pitchlist),
                      torch.tensor(enunciationlist),
                      torch.tensor(feedback_list),
                      torch.tensor(user_list)],
                     dim=1)

# Define your targets as PyTorch tensors
# targets = torch.tensor(feedback_list)

# Define your batch size
# batch_size = 16
batch_size = 1

model = PolicyModel(input_size=6, hidden_size=20)
criterion_reward = RewardFunction()
optimizer = optim.Adam(model.parameters(), lr=0.001)


for i in range(0, inputs.size(0), batch_size):
    # print("i:",i)
    batch_inputs = inputs[i:i + batch_size]
    # batch_targets = targets[i:i + batch_size]

    # reward = criterion_reward()
    # cost = criterion_cost(torch.tensor(float(feedback)), torch.tensor(y_pred_feedback))
    # optimizer.zero_grad()
    # loss = reward
    # loss = Variable(loss, requires_grad=True)

    # Forward pass
    # output = torch.matmul(input_tensor.transpose(0, 1), weights)
    # batch_inputs = batch_inputs.transpose(0, 1)
    outputs = model(batch_inputs)

    # user = outputslist[-1]
    # correct_rate = outputslist[-2]
    # [0.8988415002822876', ' 0.10115855932235718', ' 0.8261094093322754', ' 0.17389057576656342', ' 0.023385751992464066
    #  ', ' 0.9766142964363098', ' 0.005386887118220329', ' 0.9946131110191345', ' 1.0', ' 1.0]
    # ' new: [[0.39720773696899414', ' 253.84432983398438', ' 2.0054686069488525', ' 0.25928282737731934',' 0.5515929460525513', ' 5.489624977111816]', \
    #   ' [3.07273669941481e-14', ' 397.2736511230469', ' 3.3850104808807373', ' 0.5', ' 1.0', ' 4.56477303872882e-12]', \
    #   ' [0.2935240566730499', ' 259.9603576660156', ' 2.0310091972351074', ' 0.32783228158950806', ' 0.6404625177383423', ' 4.419103145599365]',
    #   ' [0.39720773696899414', ' 253.84432983398438', ' 2.0054686069488525', ' 0.25928282737731934', ' 0.5515929460525513', ' 5.489624977111816]',\
    #   ' [0.4095400869846344', ' 253.15773010253906', ' 2.0026302337646484', ' 0.2513740658760071', ' 0.541397213935852', ' 5.607595920562744]', \
    #   ' [0.4095400869846344', ' 253.15773010253906', ' 2.0026302337646484', ' 0.2513740658760071', ' 0.541397213935852', ' 5.607595920562744]]'

    # Compute loss
    # loss = criterion(outputs, batch_targets)
    # model = PolicyModel(input_size=6, hidden_size=20)
    # criterion_reward = RewardFunction()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    reward = criterion_reward()
    loss = reward
    loss = Variable(loss, requires_grad=True)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.sum().backward()
    optimizer.step()

    outputslist = outputs.tolist()[0]
    voiceoutput = outputslist[:-2]
    action, probality = select_action(voiceoutput)

# print('ap:', action)
# print('batch_inputs:', batch_inputs)
print('new:', i, voiceoutput)
# # number=i


VOLUME_RANGE[0], VOLUME_RANGE[1], index_increcement_1 = Trustregion(volumelist, VOLUME_RANGE)
SPEED_RANGE[0], SPEED_RANGE[1], index_increcement_2 = Trustregion(speedlist, SPEED_RANGE)
PITCH_RANGE[0], PITCH_RANGE[1], index_increcement_3 = Trustregion(pitchlist, PITCH_RANGE)
ENUNCIATION_RANGE[0], ENUNCIATION_RANGE[1], index_increcement_4 = Trustregion(enunciationlist, ENUNCIATION_RANGE)



theta = [0.2,20,0.4,0.08]
for i in range(len(theta)):
    theta[i] /= index_increcement_1

# print('index:',index_increcement_1)

if feedback < 1:
    theta = [0.2, 20, 0.4, 0.08]
    if VOLUME_RANGE[1] != 1 and VOLUME_RANGE[1] != 0.1:
        VOLUME_RANGE = [VOLUME_RANGE[1], 1]
    else:
        VOLUME_RANGE = [0.1, 1]

    if SPEED_RANGE[1] != 400 and SPEED_RANGE[1] != 50:
        SPEED_RANGE = [50, SPEED_RANGE[1]]
    else:
        SPEED_RANGE = [50, 400]

    if PITCH_RANGE[1] != 4 and PITCH_RANGE[1] != 1 :
        PITCH_RANGE = [1, PITCH_RANGE[1]]
    else:
        PITCH_RANGE = [1, 4]

    if ENUNCIATION_RANGE[1] != 0.5 and ENUNCIATION_RANGE[1] != 0.01 :
        ENUNCIATION_RANGE = [ENUNCIATION_RANGE[1] , 0.5]
    else:
        ENUNCIATION_RANGE = [0.01, 0.5]
# Update the last state of parameters
# new_parameters = [last_parameters[i] + index[i]*y_pred[i] for i in range(4)]

new_parameters_1 = last_parameters[0] + theta[0]*action[0]
new_parameters_2 = last_parameters[1] + theta[1]*action[1]
new_parameters_3 = last_parameters[2] + theta[2]*action[2]
new_parameters_4 = last_parameters[3] + theta[3]*action[3]
new_parameters = [new_parameters_1, new_parameters_2, new_parameters_3, new_parameters_4]

last_parameters = new_parameters
# print('last_parameters:',last_parameters)


# # Update the model with the feedback
# reward = criterion_reward()
# # cost = criterion_cost(torch.tensor(float(feedback)), torch.tensor(y_pred_feedback))
# optimizer.zero_grad()
# loss = reward * sum(probality)
# loss = Variable(loss, requires_grad=True)
# loss.sum().backward()
# optimizer.step()


# print('new_parameters: ', new_parameters)
# Update the voice parameters with the new values
Volume, speed, pitch, enunciation = new_parameters

new_parameters[0] = min(max(Volume, VOLUME_RANGE[0]), VOLUME_RANGE[1])
new_parameters[1] = min(max(speed, SPEED_RANGE[0]), SPEED_RANGE[1])
new_parameters[2] = min(max(pitch, PITCH_RANGE[0]), PITCH_RANGE[1])
new_parameters[3] = min(max(enunciation, ENUNCIATION_RANGE[0]), ENUNCIATION_RANGE[1])
# print('new_parameters 2: ', new_parameters)

# print('new_parameters:',new_parameters)
sys.stdout.write(str(new_parameters))
sys.stdout.flush()
