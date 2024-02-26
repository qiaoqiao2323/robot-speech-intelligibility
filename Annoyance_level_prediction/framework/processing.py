import torch
import torch.nn.functional as F
import numpy as np

from framework.models_pytorch import move_data_to_gpu


def forward_asc_aec(model, generate_func, cuda):
    outputs = []
    outputs_events = []
    all_ids = []

    # Evaluate on mini-batch
    for data in generate_func:
        (batch_x, batch_ids) = data

        all_ids.extend(batch_ids)

        batch_x = move_data_to_gpu(batch_x, cuda)
        # print(batch_x.size())

        model.eval()
        with torch.no_grad():
            linear_each_events, linear_rate = model(batch_x)
            batch_each_events = F.sigmoid(linear_each_events)
            outputs_events.append(batch_each_events.data.cpu().numpy())
            outputs.append(linear_rate.data.cpu().numpy())

    dict = {}

    if len(outputs):
        outputs = np.concatenate(outputs, axis=0)
    dict['output'] = outputs

    outputs_events = np.concatenate(outputs_events, axis=0)
    dict['outputs_events'] = outputs_events

    dict['all_ids'] = all_ids
    return dict










