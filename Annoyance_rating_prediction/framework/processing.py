import time, os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from framework.utilities import create_folder
from framework.models_pytorch import move_data_to_gpu
import framework.config as config
from sklearn import metrics


def define_system_name(alpha=None, basic_name='system', att_dim=None, n_heads=None,
                       batch_size=None, epochs=None):
    suffix = ''
    if alpha:
        suffix = suffix.join([str(each) for each in alpha]).replace('.', '')

    sys_name = basic_name
    sys_suffix = '_b' + str(batch_size) + '_e' + str(epochs) \
                 + '_attd' + str(att_dim) + '_h' + str(n_heads) if att_dim is not None and n_heads is not None \
        else '_b' + str(batch_size)  + '_e' + str(epochs)

    sys_suffix = sys_suffix + '_cuda' + str(config.cuda_seed) if config.cuda_seed is not None else sys_suffix
    system_name = sys_name + sys_suffix if sys_suffix is not None else sys_name

    return suffix, system_name


def forward_asc_aec(model, generate_func, cuda):
    outputs = []
    outputs_event = []

    targets = []
    targets_event = []

    # Evaluate on mini-batch
    for data in generate_func:
        (batch_x, batch_y, batch_y_event) = data

        batch_x = move_data_to_gpu(batch_x, cuda)

        model.eval()
        with torch.no_grad():
            all_output = model(batch_x)  # torch.Size([16, 10])
            batch_output_event, batch_rate = all_output[0], all_output[1]

            batch_output_event = F.sigmoid(batch_output_event)

            outputs.append(batch_rate.data.cpu().numpy())
            outputs_event.append(batch_output_event.data.cpu().numpy())

            targets.append(batch_y)
            targets_event.append(batch_y_event)

    dict = {}

    if len(outputs):
        outputs = np.concatenate(outputs, axis=0)
    dict['output'] = outputs

    if len(outputs_event):
        outputs_event = np.concatenate(outputs_event, axis=0)
    dict['outputs_event'] = outputs_event

    targets = np.concatenate(targets, axis=0)
    dict['target'] = targets
    targets_event = np.concatenate(targets_event, axis=0)
    dict['targets_event'] = targets_event
    return dict


def forward_asc_aec_only_rms(model, generate_func, cuda):
    outputs = []
    outputs_event = []

    targets = []
    targets_event = []

    # Evaluate on mini-batch
    for data in generate_func:
        (batch_x, batch_x_rms, batch_y, batch_y_event) = data

        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_x_rms = move_data_to_gpu(batch_x_rms, cuda)
        # print(batch_x.size())

        model.eval()
        with torch.no_grad():
            all_output = model(batch_x_rms)  # torch.Size([16, 10])
            batch_rate, batch_output_event = all_output[0], all_output[1]

            batch_output_event = F.sigmoid(batch_output_event)

            outputs.append(batch_rate.data.cpu().numpy())
            outputs_event.append(batch_output_event.data.cpu().numpy())

            targets.append(batch_y)
            targets_event.append(batch_y_event)

    dict = {}

    if len(outputs):
        outputs = np.concatenate(outputs, axis=0)
    dict['output'] = outputs

    if len(outputs_event):
        outputs_event = np.concatenate(outputs_event, axis=0)
    dict['outputs_event'] = outputs_event

    targets = np.concatenate(targets, axis=0)
    dict['target'] = targets
    targets_event = np.concatenate(targets_event, axis=0)
    dict['targets_event'] = targets_event
    return dict

def forward_asc_aec_only_mel(model, generate_func, cuda):
    outputs = []
    outputs_event = []

    targets = []
    targets_event = []

    # Evaluate on mini-batch
    for data in generate_func:
        (batch_x, batch_x_rms, batch_y, batch_y_event) = data

        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_x_rms = move_data_to_gpu(batch_x_rms, cuda)
        # print(batch_x.size())

        model.eval()
        with torch.no_grad():
            all_output = model(batch_x)  # torch.Size([16, 10])
            batch_rate, batch_output_event = all_output[0], all_output[1]

            batch_output_event = F.sigmoid(batch_output_event)

            outputs.append(batch_rate.data.cpu().numpy())
            outputs_event.append(batch_output_event.data.cpu().numpy())

            targets.append(batch_y)
            targets_event.append(batch_y_event)

    dict = {}

    if len(outputs):
        outputs = np.concatenate(outputs, axis=0)
    dict['output'] = outputs

    if len(outputs_event):
        outputs_event = np.concatenate(outputs_event, axis=0)
    dict['outputs_event'] = outputs_event

    targets = np.concatenate(targets, axis=0)
    dict['target'] = targets
    targets_event = np.concatenate(targets_event, axis=0)
    dict['targets_event'] = targets_event
    return dict


def forward_asc_aec_event_emb(model, generate_func, cuda):
    outputs = []
    outputs_event = []

    targets = []
    targets_event = []

    batch_event_emb_list = []

    # Evaluate on mini-batch
    for data in generate_func:
        (batch_x, batch_x_rms, batch_y, batch_y_event) = data

        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_x_rms = move_data_to_gpu(batch_x_rms, cuda)
        # print(batch_x.size())

        model.eval()
        with torch.no_grad():
            all_output = model(batch_x, batch_x_rms, True)  # torch.Size([16, 10])
            batch_rate, batch_output_event, batch_event_emb = all_output[0], all_output[1], all_output[2]

            batch_output_event = F.sigmoid(batch_output_event)

            batch_event_emb_list.append(batch_event_emb.data.cpu().numpy())

            outputs.append(batch_rate.data.cpu().numpy())
            outputs_event.append(batch_output_event.data.cpu().numpy())

            targets.append(batch_y)
            targets_event.append(batch_y_event)

    dict = {}

    if len(outputs):
        outputs = np.concatenate(outputs, axis=0)
    dict['output'] = outputs

    if len(outputs_event):
        outputs_event = np.concatenate(outputs_event, axis=0)
    dict['outputs_event'] = outputs_event

    batch_event_emb_list = np.concatenate(batch_event_emb_list, axis=0)
    dict['batch_event_emb_list'] = batch_event_emb_list

    targets = np.concatenate(targets, axis=0)
    dict['target'] = targets
    targets_event = np.concatenate(targets_event, axis=0)
    dict['targets_event'] = targets_event
    return dict


def forward_asc_aec_return_atts(model, generate_func, cuda):
    outputs = []
    outputs_event = []

    targets = []
    targets_event = []

    q_rms_kv_mel_atts_list, q_mel_kv_rms_atts_list = [], []

    # Evaluate on mini-batch
    for data in generate_func:
        (batch_x, batch_x_rms, batch_y, batch_y_event) = data

        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_x_rms = move_data_to_gpu(batch_x_rms, cuda)
        # print(batch_x.size())

        model.eval()
        with torch.no_grad():
            all_output = model(batch_x, batch_x_rms)  # torch.Size([16, 10])
            batch_rate, batch_output_event, q_rms_kv_mel_atts, q_mel_kv_rms_atts = all_output[0], all_output[1], \
                                                                                   all_output[2], all_output[3]

            q_rms_kv_mel_atts_list.append(q_rms_kv_mel_atts.data.cpu().numpy())
            q_mel_kv_rms_atts_list.append(q_mel_kv_rms_atts.data.cpu().numpy())

            batch_output_event = F.sigmoid(batch_output_event)

            outputs.append(batch_rate.data.cpu().numpy())
            outputs_event.append(batch_output_event.data.cpu().numpy())

            targets.append(batch_y)
            targets_event.append(batch_y_event)

    dict = {}

    if len(outputs):
        outputs = np.concatenate(outputs, axis=0)
    dict['output'] = outputs

    if len(outputs_event):
        outputs_event = np.concatenate(outputs_event, axis=0)
    dict['outputs_event'] = outputs_event

    q_mel_kv_rms_atts_list = np.concatenate(q_mel_kv_rms_atts_list, axis=0)
    dict['q_mel_kv_rms_atts_list'] = q_mel_kv_rms_atts_list
    q_rms_kv_mel_atts_list = np.concatenate(q_rms_kv_mel_atts_list, axis=0)
    dict['q_rms_kv_mel_atts_list'] = q_rms_kv_mel_atts_list

    # print(q_rms_kv_mel_atts_list.shape, q_mel_kv_rms_atts_list.shape)
    # (578, 8, 30, 30) (578, 8, 30, 30)

    targets = np.concatenate(targets, axis=0)
    dict['target'] = targets
    targets_event = np.concatenate(targets_event, axis=0)
    dict['targets_event'] = targets_event
    return dict


def evaluate_asc_aec(model, generator, data_type, cuda):

    # Generate function
    generate_func = generator.generate_validate(data_type=data_type)

    # Forward
    dict = forward_asc_aec(model=model, generate_func=generate_func, cuda=cuda)

    # rate loss
    targets = dict['output']
    predictions = dict['target']
    # rate_mse_loss = metrics.mean_squared_error(targets, predictions)
    rate_mse_loss = metrics.mean_squared_error(targets, predictions, squared=False)
    # rmse

    # aec
    outputs_event = dict['outputs_event']  # (audios_num, classes_num)
    targets_event = dict['targets_event']  # (audios_num, classes_num)
    aucs = []
    for i in range(targets_event.shape[0]):
        test_y_auc, pred_auc = targets_event[i, :], outputs_event[i, :]
        if np.sum(test_y_auc):
            test_auc = metrics.roc_auc_score(test_y_auc, pred_auc)
            aucs.append(test_auc)
    final_auc = sum(aucs) / len(aucs)

    return rate_mse_loss, final_auc

